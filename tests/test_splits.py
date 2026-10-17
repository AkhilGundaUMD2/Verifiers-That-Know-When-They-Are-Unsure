import unittest
from collections import defaultdict
from loaders.splits import assign_splits


def record(record_id, group_id, domain):
    # Minimal processed record; labels are irrelevant to split assignment.
    return {'id': record_id, 'group_id': group_id, 'domain': domain, 'labels': {}}


def dataset():
    # Two agents per task across two in-distribution domains and one OOD domain.
    return [record(f'{domain}/{agent}/{task}', f'{domain}.{task}', domain)
            for domain in ('webarena', 'workarena', 'assistantbench')
            for task in range(10) for agent in ('a', 'b')]


class SplitTests(unittest.TestCase):
    def test_ood_domain_is_held_out(self):
        # Every held-out record goes to ood_test and nothing else does.
        records = dataset()
        assignment = assign_splits(records, ['assistantbench'])
        for r in records:
            self.assertEqual(assignment[r['id']] == 'ood_test', r['domain'] == 'assistantbench')

    def test_groups_stay_together(self):
        # Trajectories of the same task by different agents share one split.
        records = dataset()
        assignment = assign_splits(records, ['assistantbench'])
        splits = defaultdict(set)
        for r in records:
            splits[r['group_id']].add(assignment[r['id']])
        self.assertTrue(all(len(s) == 1 for s in splits.values()))

    def test_fraction_and_determinism(self):
        # 20 in-distribution groups at 0.25 give 5 calibration groups (10 records),
        # and the result does not depend on input order.
        records = dataset()
        assignment = assign_splits(records, ['assistantbench'], 0.25, seed=3)
        self.assertEqual(list(assignment.values()).count('calibration'), 10)
        self.assertEqual(assignment, assign_splits(list(reversed(records)), ['assistantbench'], 0.25, seed=3))

    def test_seed_changes_assignment(self):
        records = dataset()
        self.assertNotEqual(assign_splits(records, ['assistantbench'], seed=0),
                            assign_splits(records, ['assistantbench'], seed=1))

    def test_unknown_ood_domain_fails(self):
        with self.assertRaisesRegex(ValueError, 'not present'):
            assign_splits(dataset(), ['missing'])

    def test_group_crossing_ood_boundary_fails(self):
        # A task group appearing in both an OOD and an in-distribution domain would leak.
        records = dataset() + [record('extra', 'webarena.0', 'assistantbench')]
        with self.assertRaisesRegex(ValueError, 'spans'):
            assign_splits(records, ['assistantbench'])

    def test_duplicate_record_fails(self):
        records = dataset()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            assign_splits(records + records[:1], ['assistantbench'])


if __name__ == '__main__':
    unittest.main()
