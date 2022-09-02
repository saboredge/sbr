#!/usr/bin/env python
import shutil
import os
shutil.rmtree('tests/out', ignore_errors=True, onerror=None)
os.makedirs('tests/out')
import sys
sys.path.insert(0, f'.')
from unittest import TestCase

import numpy as np
import difflib
import tensorflow as tf

import monkey_patch

class TestSBR(TestCase):

    def fcmp(self, test_file_name):
        import filecmp
        self.assertTrue(filecmp.cmp(f'tests/out/{test_file_name}',
                                    f'tests/expected/{test_file_name}'))

    def cache_load(self):
        if monkey_patch.LOAD_DONE:
            return
        # Test 1 -----------
        import tensorflow as tf
        import tensorflow_datasets as tfds
        from sbr.datasets.structured import gtex
        monkey_patch.ds, monkey_patch.info = tfds.load("gtex", split="train", with_info = True,as_supervised =True)
        l=list(iter(monkey_patch.ds.take(monkey_patch.info.splits['train'].num_examples)))
        monkey_patch.X, monkey_patch.y = map(np.array, zip(*l))
        monkey_patch.LOAD_DONE = True

    def test_load(self):
        self.cache_load()
        self.assertTrue(monkey_patch.y[0] == 11)


    def cache_setup(self):
        if monkey_patch.SETUP_DONE:
            return

        import sbr.preprocessing.gtex
        [monkey_patch.class_names,
         monkey_patch.x_train, monkey_patch.y_train, 
         monkey_patch.x_validation, monkey_patch.y_validation, 
         monkey_patch.x_test,monkey_patch.y_test] = sbr.preprocessing.gtex.dataset_setup(sample_count_threshold=500, 
                                                               test_fraction = 0.1, 
                                                               validation_fraction = 0.1, 
                                                               verbose = True, 
                                                               batch_size = 32, 
                                                               seed = 42)
        monkey_patch.SETUP_DONE = True




    def test_setup_classnames(self):
        # Test 2 -----------
        self.cache_setup()
        self.assertTrue((monkey_patch.class_names == monkey_patch.class_names_expected).all())

    def test_setup_first_label(self):
        # Test 3 -----------
        self.cache_setup()
        y_train_0_expected = [0.] * 30
        #y_train_0_expected[16] = 1.
        y_train_0_expected[0] = 1.
        y_train_0_expected = np.array(y_train_0_expected, dtype=float)

        print(f"y_train[0] = {monkey_patch.y_train[0]}")
        self.assertTrue((monkey_patch.y_train[0] == y_train_0_expected).all())

    def cache_compile(self):
        if monkey_patch.COMPILE_DONE:
            return
        from sbr import compile
        specificityAtSensitivityThreshold=0.50
        sensitivityAtSpecificityThreshold=0.50
        monkey_patch.model = compile.one_layer_multicategorical(input_size=monkey_patch.x_train.shape[1],
                                                        output_size=monkey_patch.y_train.shape[1],
                                                        dim=1000,
                                                        output_activation='softmax',
                                                        learning_rate=0.0001,
                                                        isMultilabel=True,
                                                        specificityAtSensitivityThreshold=specificityAtSensitivityThreshold,
                                                        sensitivityAtSpecificityThreshold=sensitivityAtSpecificityThreshold,
                                                        kernel_initializer = tf.keras.initializers.HeNormal(seed = 42), 
                                                        bias_initializer = tf.zeros_initializer(),
                                                        seed=42,
                                                        verbose=True)
        monkey_patch.COMPILE_DONE = True
        
    def cache_fit(self):
        if monkey_patch.FIT_DONE:
            return
        from sbr import fit
        monkey_patch.history=fit.multicategorical_model(model=monkey_patch.model,
                                                model_folder ='tests/out',
                                                x_train=monkey_patch.x_train, y_train=monkey_patch.y_train,
                                                x_validation=monkey_patch.x_validation, y_validation=monkey_patch.y_validation,
                                                epochs = 3,
                                                patience = 2,
                                                lr_patience = 1,
                                                checkpoint_verbose=1,
                                                train_verbose=0,
                                                seed = 42)
        monkey_patch.FIT_DONE = True

    def test_compile(self):
        self.cache_setup()
        self.cache_compile()

        with open('tests/out/compile.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            monkey_patch.model.summary(print_fn=lambda x: fh.write(x + '\n'))

        self.fcmp('compile.txt')

    def write_test_pred_pairs(self, test_filename, pairs=None, write_tests=False):
        import json
        from json import JSONEncoder
        class NumpyArrayEncoder(JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return JSONEncoder.default(self, obj)

        with open(f"tests/out/{test_filename}", "a") as f:
            y_pred_json = json.dumps(np.argmax(monkey_patch.y_pred,axis=1).astype(int),
                                     cls=NumpyArrayEncoder) 
            f.write("{  ")
            
            if write_tests:
                y_test_json = json.dumps(np.argmax(monkey_patch.y_test,axis=1).astype(int), cls=NumpyArrayEncoder) 
                f.write(f"  'y_test': {y_test_json}\n")

            # print pred no matter what
            y_pred_json = json.dumps(np.argmax(monkey_patch.y_pred,axis=1).astype(int), cls=NumpyArrayEncoder) 
            f.write(f"  'y_pred': {y_pred_json}\n")

            if pairs is not None:
                f.write("  'pairs': [  ")
                for line in pairs:
                    f.write(f"{line}, ")
                f.write("  ]\n")

            f.write("}\n")


    def test_pred(self):
        self.cache_setup()
        self.cache_compile()
        self.cache_fit()

        monkey_patch.y_pred = monkey_patch.model.predict(monkey_patch.x_test) # xxx
        self.write_test_pred_pairs("pred.json")

        self.fcmp("pred.json")


    def test_evaluate_compare_pred(self):
        self.cache_setup()
        self.cache_compile()
        self.cache_fit()
        from sbr.evaluate import compare_predictions
        monkey_patch.y_pred, pairs = compare_predictions(y_pred=monkey_patch.model.predict(monkey_patch.x_test),
                                                 y_test=monkey_patch.y_test,
                                                 class_names=monkey_patch.class_names,
                                                 verbose = True)

        self.write_test_pred_pairs("evaluate_compare_pred.json", pairs=pairs)
        self.fcmp("evaluate_compare_pred.json")
        


    # xxx do this next
    '''
    def test_evaluate_mislabeled(self):
        self.cache_setup()
        self.cache_compile()
        self.cache_fit()
        # xxx where to get label_df?
        mislabeled_counts, mislabeled = mislabeled_pair_counts(model=monkey_patch.model, X=monkey_patch.X, y=monkey_patch.y, class_names=monkey_patch.class_names,
                                                               sample_ids = pd.Series(label_df["sample_id"]),
                                                               batch_size=500)
        print(mislabeled_counts)
    '''

