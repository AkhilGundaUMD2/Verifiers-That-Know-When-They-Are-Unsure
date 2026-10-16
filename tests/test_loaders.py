import csv
import json
import tempfile
import unittest
from pathlib import Path
from loaders.benchmarks import load_agentrewardbench, load_osworld


class LoaderTests(unittest.TestCase):
    def setUp(self):
        # give each test its own temporary dataset directory and remove it afterward.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def arb(self, labels):
        # Create one AgentRewardBench trajectory with a single recorded click.
        path = self.root / 'cleaned' / 'webarena' / 'agent' / 'exp' / 'webarena.1.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'benchmark': 'webarena', 'agent': 'agent', 'goal': 'Task', 'steps': [{'action': 'click'}]}))
        # Add one annotation row per (success, side-effect) pair supplied by the
        # test. Every row marks looping as No; an empty list writes only a header.
        annotations = self.root / 'annotations.csv'
        fields = ['benchmark', 'model_name', 'task_id', 'trajectory_success', 'trajectory_side_effect', 'trajectory_looping']
        with annotations.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for success, side in labels:
                writer.writerow(dict(zip(fields, ['webarena', 'agent', 'webarena.1', success, side, 'No'])))
        return annotations

    def test_join_and_preserve_steps(self):
        # check that CSV annotations join to the trajectory, labels become
        # booleans, the recorded action is preserved, and the task group is correct.
        annotation = self.arb([('Successful', 'Yes')])
        record = list(load_agentrewardbench(self.root, annotation))[0]
        self.assertEqual(record['labels'], {'success': True, 'side_effect': True, 'looping': False})
        self.assertEqual(record['steps'], [{'action': 'click'}])
        self.assertEqual(record['group_id'], 'agentrewardbench/webarena.1')

    def test_disagreement_retains_both_annotations(self):
        # conflicting success judgments should produce an unknown label (None)
        # while keeping both original annotations for inspection.
        annotation = self.arb([('Successful', 'No'), ('Unsuccessful', 'No')])
        record = list(load_agentrewardbench(self.root, annotation))[0]
        self.assertIsNone(record['labels']['success'])
        self.assertEqual(len(record['annotations']), 2)

    def test_missing_labels_are_unknown(self):
        # a trajectory with no annotation rows should have an unknown side-effect
        # label rather than defaulting to False.
        annotation = self.arb([])
        record = list(load_agentrewardbench(self.root, annotation))[0]
        self.assertIsNone(record['labels']['side_effect'])

    def test_unsure_is_unknown(self):
        # Explicit Unsure judgments should map to None for success and side effects.
        annotation = self.arb([('Unsure', 'Unsure')])
        record = list(load_agentrewardbench(self.root, annotation))[0]
        self.assertIsNone(record['labels']['success'])
        self.assertIsNone(record['labels']['side_effect'])

    def test_unmatched_annotation_fails(self):
        # Add an annotation for a task with no trajectory JSON and verify that
        # fully consuming the loader raises an error instead of silently dropping it.
        annotation = self.arb([('Successful', 'No')])
        with annotation.open('a') as stream:
            stream.write('webarena,agent,webarena.2,Successful,No,No\n')
        with self.assertRaisesRegex(ValueError, 'no matching JSON'):
            list(load_agentrewardbench(self.root, annotation))

    def osworld(self):
        # create a minimal OsWorld manifest and its matching Chrome task definition.
        (self.root / 'test_all.json').write_text(json.dumps({'chrome': ['task']}))
        folder = self.root / 'examples' / 'chrome'
        folder.mkdir(parents=True)
        (folder / 'task.json').write_text(json.dumps({'id': 'task', 'instruction': 'Task', 'evaluator': {'func': 'exact_match'}, 'trajectory': 'trajectories/'}))

    def test_osworld_is_not_a_labeled_run(self):
        # check that an OsWorld task loads as a task definition with unknown steps
        # and success, while preserving its evaluator configuration.
        self.osworld()
        record = list(load_osworld(self.root))[0]
        self.assertEqual(record['record_type'], 'task_definition')
        self.assertIsNone(record['steps'])
        self.assertIsNone(record['labels']['success'])
        self.assertEqual(record['evaluator'], {'func': 'exact_match'})

    def test_missing_osworld_task_fails(self):
        # delete a task referenced by the manifest and expect FileNotFoundError.
        self.osworld()
        (self.root / 'examples/chrome/task.json').unlink()
        with self.assertRaises(FileNotFoundError):
            list(load_osworld(self.root))

    def test_duplicate_osworld_task_fails(self):
        # list the same task twice in the manifest and verify duplicate IDs are rejected.
        self.osworld()
        (self.root / 'test_all.json').write_text(json.dumps({'chrome': ['task', 'task']}))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            list(load_osworld(self.root))


if __name__ == '__main__':
    # Run this test suite when the file is executed directly.
    unittest.main()
