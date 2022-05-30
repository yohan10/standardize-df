import unittest
import doctest
from standardize_df import pipeline
from standardize_df.pipeline import PipelineMapping, Pipeline
from collections import OrderedDict


def foo(): pass


class PipelineSubclass(Pipeline):
    pass


class TestPipeline(unittest.TestCase):
    def test_only_callables_allowed(self):
        # Non callable values passed in raises TypeError on...
        # 1. __init__
        self.assertRaises(TypeError, Pipeline, [3, 4])

        # 2. insert
        pipeline = Pipeline([])  # Empty pipeline should be allowed
        self.assertRaises(TypeError, pipeline.insert, 0, 3)

        # 3. __setitem__
        pipeline = Pipeline([foo])
        self.assertRaises(TypeError, pipeline.__setitem__, 0, 'sewage')

        def implicit_setitem(pipeline):
            pipeline[0] = 'sewage'
        self.assertRaises(TypeError, implicit_setitem, pipeline)

    def test_add(self):
        pipeline = Pipeline([foo])
        self.assertRaises(TypeError, lambda: pipeline + [foo])
        self.assertEqual(
            pipeline + pipeline,
            Pipeline([foo, foo])
        )
        pipeline_sub = PipelineSubclass([foo])
        self.assertEqual(
            pipeline + pipeline_sub,
            Pipeline([foo, foo])
        )
        self.assertEqual(type(pipeline + pipeline_sub), Pipeline)

    def test_radd(self):
        pipeline = Pipeline([foo])
        pipeline_sub = PipelineSubclass([foo])
        self.assertRaises(TypeError, lambda: [foo] + pipeline)
        self.assertEqual(
            pipeline_sub + pipeline,
            PipelineSubclass([foo, foo])
        )
        self.assertEqual(type(pipeline_sub + pipeline), PipelineSubclass)

    def test_iadd(self):
        pipeline = Pipeline([foo])
        pipeline_id = id(pipeline)
        pipeline += (foo, )
        self.assertEqual(pipeline, Pipeline([foo, foo]))
        self.assertEqual(pipeline_id, id(pipeline))
        pipeline += pipeline
        self.assertEqual(pipeline, Pipeline([foo, foo, foo, foo]))
        self.assertEqual(type(pipeline), Pipeline)

        def augment_assign_non_iter(pipeline):
            pipeline += 3
        self.assertRaises(TypeError, augment_assign_non_iter, pipeline)

    def test_mul(self):
        pipeline = Pipeline([foo])
        pipeline_id = id(pipeline)
        self.assertRaises(TypeError, lambda: pipeline * 1.0)
        self.assertRaises(TypeError, lambda: pipeline * pipeline)
        self.assertEqual(pipeline * 2, Pipeline([foo, foo]))
        self.assertNotEqual(pipeline_id, id(pipeline * 2))

    def test_rmul(self):
        pipeline = Pipeline([foo])
        pipeline_id = id(pipeline)
        pipeline_sub = PipelineSubclass([foo])
        self.assertRaises(TypeError, lambda: 1.0 * pipeline)
        self.assertRaises(TypeError, lambda: pipeline_sub * pipeline)
        self.assertEqual(2 * pipeline, Pipeline([foo, foo]))
        self.assertNotEqual(pipeline_id, id(2 * pipeline))

    def test_imul(self):
        pipeline = Pipeline([foo])
        pipeline_id = id(pipeline)

        def augment_assign_non_int(pipeline, non_int):
            pipeline *= non_int
        self.assertRaises(TypeError, augment_assign_non_int, pipeline, 1.0)
        self.assertRaises(TypeError, augment_assign_non_int, pipeline, pipeline)

        pipeline *= 2
        self.assertEqual(pipeline, Pipeline([foo, foo]))
        self.assertEqual(pipeline_id, id(pipeline))
        self.assertEqual(type(pipeline), Pipeline)

        pipeline_sub = PipelineSubclass([foo])
        pipeline_sub *= 2
        self.assertEqual(type(pipeline), Pipeline)

    def test_eq(self):
        self.assertEqual(
            Pipeline([foo, foo]),
            Pipeline([foo, foo])
        )
        self.assertNotEqual(
            Pipeline([foo, foo]),
            Pipeline([foo, foo, foo])
        )
        def fu(): pass
        self.assertNotEqual(
            Pipeline([foo, foo]),
            Pipeline([foo, fu])
        )

    def test_call(self):
        def square(x):
            return x ** x

        pipeline = Pipeline([square, square])
        self.assertEqual(pipeline(2), 256)

        # test that runs in order.
        def add_hello(string):
            return 'Hello ' + string + '!'

        def add_smooth_compliment(string):
            return string + " Your weather is looking nice."

        pipeline = Pipeline([add_hello, add_smooth_compliment])

        self.assertEqual(
            pipeline('Beyonce'),
            'Hello Beyonce! Your weather is looking nice.'
        )

    def test_custom_pipelines_enforces_types(self):
        class StrIntPipeline(Pipeline):
            enforce_types = (str, int)

        def multiply_two(obj):
            return obj * 2

        def multiply_three(obj):
            return obj * 3

        def return_dict(obj):
            return {obj: None}

        pipeline = StrIntPipeline([multiply_two, multiply_three])

        self.assertEqual(pipeline(2), 12)
        self.assertEqual(pipeline('a'), 'aaaaaa')

        # initial object {} raises TypeError if not in PipelineMapping.enforced_types.
        self.assertRaises(TypeError, pipeline, {})
        try:
            pipeline({})
        except TypeError as err:
            keywords = [
                "obj",
                "not match",
                "enforced types",
                "(<class 'str'>, <class 'int'>)",
                "got: <class 'dict'>"
            ]
            err_msg = str(err)
            for k in keywords:
                self.assertIn(k, err_msg)

        pipeline.append(return_dict)

        # Results from a pipeline operation raises TypeError if not in
        # Pipeline.enforced_types.
        self.assertRaises(TypeError, pipeline, 2)
        try:
            pipeline(2)
        except TypeError as err:
            keywords = [
                "Result",
                "not match",
                "enforced types",
                "(<class 'str'>, <class 'int'>)",
                "got: <class 'dict'>"
            ]
            err_msg = str(err)
            for k in keywords:
                self.assertIn(k, err_msg)


class TestPipelineMapping(unittest.TestCase):
    def test_pipeline_state_is_ordered_dict(self):
        def foo(): pass
        pipeline = PipelineMapping({'a': foo})
        self.assertIsInstance(pipeline._pipeline, OrderedDict)

    def test_call(self):
        def square(x):
            return x**x

        pipeline = PipelineMapping({'square': square})
        self.assertEqual(pipeline(3), 27)

        # test that runs in order.
        def add_hello(string):
            return 'Hello ' + string + '!'

        def add_smooth_compliment(string):
            return string + " Your weather is looking nice."

        pipeline = PipelineMapping(
            add_hello=add_hello,
            smooth_compliment=add_smooth_compliment
        )

        self.assertEqual(
            pipeline('Beyonce'),
            'Hello Beyonce! Your weather is looking nice.'
        )

    def test_custom_pipelines_enforces_types(self):
        class StrIntPipelineMapping(PipelineMapping):
            enforce_types = (str, int)

        def multiply_two(obj):
            return obj * 2

        def multiply_three(obj):
            return obj * 3

        def return_dict(obj):
            return {obj: None}

        pipeline = StrIntPipelineMapping({
            'multi_two': multiply_two,
            'multi_three': multiply_three
        })

        self.assertEqual(pipeline(2), 12)
        self.assertEqual(pipeline('a'), 'aaaaaa')

        # initial object {} raises TypeError if not in PipelineMapping.enforced_types.
        self.assertRaises(TypeError, pipeline, {})
        try:
            pipeline({})
        except TypeError as err:
            keywords = [
                "obj",
                "not match",
                "enforced types",
                "(<class 'str'>, <class 'int'>)",
                "got: <class 'dict'>"
            ]
            err_msg = str(err)
            for k in keywords:
                self.assertIn(k, err_msg)

        pipeline['return_dict'] = return_dict

        # Results from a pipeline operation raises TypeError if not in
        # PipelineMapping.enforced_types.
        self.assertRaises(TypeError, pipeline, 2)
        try:
            pipeline(2)
        except TypeError as err:
            keywords = [
                "Result",
                "not match",
                "enforced types",
                "(<class 'str'>, <class 'int'>)",
                "got: <class 'dict'>"
            ]
            err_msg = str(err)
            for k in keywords:
                self.assertIn(k, err_msg)

    def test_set_partial(self):
        def add(a, b):
            return a + b
        pipeline = PipelineMapping()
        pipeline.set_partial('add_three', add, b=3)
        self.assertEqual(pipeline(1), 4)
        self.assertEqual(pipeline(3), 6)

    def test_reorder(self):
        def quick_brown_fox(string):
            return string + " the quick brown fox"

        def jumps_over(string):
            return string + " jumps over"

        def lazy_dog(string):
            return string + " the lazy dog"

        def format_sentence(string):
            string = string.strip()
            string = string.capitalize()
            return string + '.'

        pangram = PipelineMapping(
            quick_brown_fox=quick_brown_fox,
            jumps_over=jumps_over,
            lazy_dog=lazy_dog,
            format_sentence=format_sentence
        )

        self.assertEqual(
            pangram(''),
            'The quick brown fox jumps over the lazy dog.'
        )

        # Re-order with iterable
        order = ['lazy_dog', 'jumps_over', 'quick_brown_fox', 'format_sentence']
        pangram = pangram.reorder(order)
        self.assertEqual(
            pangram(''),
            'The lazy dog jumps over the quick brown fox.'
        )

        # Re-order with dict
        order = {
            'lazy_dog': lazy_dog,
            'jumps_over': jumps_over,
            'quick_brown_fox': quick_brown_fox,
            'format_sentence': format_sentence
        }
        pangram = pangram.reorder(order)
        self.assertEqual(
            pangram(''),
            'The lazy dog jumps over the quick brown fox.'
        )

        # Re-order with None values adds funcs existing in pipeline.
        order = {
            'lazy_dog': None,
            'jumps_over': None,
            'quick_brown_fox': None,
            'format_sentence': None
        }
        pangram = pangram.reorder(order)
        self.assertEqual(
            pangram(''),
            'The lazy dog jumps over the quick brown fox.'
        )

        # Re-order and override operations.
        def quick_brown_foxzilla(string):
            return string + " the quick brown foxzilla"

        order = {
            'lazy_dog': None,
            'jumps_over': None,
            'quick_brown_fox': quick_brown_foxzilla,
            'format_sentence': None
        }
        foxzilla_pangram = pangram.reorder(order)
        self.assertEqual(
            foxzilla_pangram(''),
            'The lazy dog jumps over the quick brown foxzilla.'
        )

        # Omit operations in reorder with iterable
        order = ['quick_brown_fox', 'jumps_over', 'format_sentence']
        pangram = pangram.reorder(order)
        self.assertEqual(
            pangram(''),
            'The quick brown fox jumps over.'
        )

        # Omit operations in reorder with dict
        order = {
            'quick_brown_fox': None,
            'jumps_over': None,
            'format_sentence': None
        }
        pangram = pangram.reorder(order)
        self.assertEqual(
            pangram(''),
            'The quick brown fox jumps over.'
        )

        # Trying to re-order an operation not existing in pipeline raises
        # KeyError.
        order_iter = ['quick_brown_fox', 'jumps_over', 'not_existing_func']
        order_dict = {
            'quick_brown_fox': None,
            'jumps_over': None,
            'not_existing_func': None
        }
        self.assertRaises(KeyError, pangram.reorder, order_iter)
        self.assertRaises(KeyError, pangram.reorder, order_dict)
        keywords = [
            "not_existing_func",
            "does not exist in the pipeline"
        ]
        try:
            pangram.reorder(order_iter)
        except KeyError as err:
            err_msg = str(err)
            for k in keywords:
                self.assertIn(k, err_msg)

        try:
            pangram.reorder(order_dict)
        except KeyError as err:
            err_msg = str(err)
            for k in keywords:
                self.assertIn(k, err_msg)

    def test_reorder_returns_pipeline(self):
        def quick_brown_fox(string):
            return string + " the quick brown fox"

        def jumps_over(string):
            return string + " jumps over"

        def lazy_dog(string):
            return string + " the lazy dog"

        def format_sentence(string):
            string = string.strip()
            string = string.capitalize()
            return string + '.'

        pangram = PipelineMapping(
            quick_brown_fox=quick_brown_fox,
            jumps_over=jumps_over,
            lazy_dog=lazy_dog,
            format_sentence=format_sentence
        )

        # Re-order with iterable
        order = ['lazy_dog', 'jumps_over', 'quick_brown_fox', 'format_sentence']
        pangram = pangram.reorder(order)
        self.assertIsInstance(pangram, PipelineMapping)


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(pipeline))
    return tests


if __name__ == '__main__':
    unittest.main()
