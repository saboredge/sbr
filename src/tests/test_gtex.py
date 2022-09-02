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

class TestSBR(TestCase):
    load_done = False
    setup_done = False
    compile_done = False
    fit_done = False

    def fcmp(self, test_file_name):
        import filecmp
        self.assertTrue(filecmp.cmp(f'tests/out/{test_file_name}',
                                    f'tests/expected/{test_file_name}'))

    def cache_load(self):
        if self.load_done:
            return
        # Test 1 -----------
        import tensorflow as tf
        import tensorflow_datasets as tfds
        from sbr.datasets.structured import gtex
        self.ds, self.info = tfds.load("gtex", split="train", with_info = True,as_supervised =True)
        l=list(iter(self.ds.take(self.info.splits['train'].num_examples)))
        self.X, self.y = map(np.array, zip(*l))

    def test_load(self):
        self.cache_load()
        self.assertTrue(self.y[0] == 11)


    def cache_setup(self):
        if self.setup_done:
            return

        self.class_names_expected = ['Colon','Heart','Blood','Vagina','Thyroid','Liver','Salivary_Gland',
                                      'Pancreas','Cervix_Uteri','Prostate','Ovary','Skin','Pituitary',
                                      'Small_Intestine','Fallopian_Tube','Adrenal_Gland','Nerve',
                                      'Adipose_Tissue','Spleen','Stomach','Muscle','Blood_Vessel','Lung',
                                      'Esophagus','Brain','Testis','Uterus','Kidney','Bladder','Breast']
        import sbr.preprocessing.gtex
        [self.class_names,
         self.x_train, self.y_train, 
         self.x_validation, self.y_validation, 
         self.x_test,self.y_test] = sbr.preprocessing.gtex.dataset_setup(sample_count_threshold=500, 
                                                               test_fraction = 0.1, 
                                                               validation_fraction = 0.1, 
                                                               verbose = True, 
                                                               batch_size = 32, 
                                                               seed = 42)
        self.setup_done = True




    def test_setup_classnames(self):
        # Test 2 -----------
        self.cache_setup()
        self.assertTrue((self.class_names == self.class_names_expected).all())

    def test_setup_first_label(self):
        # Test 3 -----------
        self.cache_setup()
        y_train_0_expected = [0.] * 30
        #y_train_0_expected[16] = 1.
        y_train_0_expected[0] = 1.
        y_train_0_expected = np.array(y_train_0_expected, dtype=float)

        print(f"y_train[0] = {self.y_train[0]}")
        self.assertTrue((self.y_train[0] == y_train_0_expected).all())

    def cache_compile(self):
        if self.compile_done:
            return
        from sbr import compile
        specificityAtSensitivityThreshold=0.50
        sensitivityAtSpecificityThreshold=0.50
        self.model = compile.one_layer_multicategorical(input_size=self.x_train.shape[1],
                                                        output_size=self.y_train.shape[1],
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
        
    def cache_fit(self):
        if self.fit_done:
            return
        from sbr import fit
        self.history=fit.multicategorical_model(model=self.model,
                                                model_folder ='tests/out',
                                                x_train=self.x_train, y_train=self.y_train,
                                                x_validation=self.x_validation, y_validation=self.y_validation,
                                                epochs = 3,
                                                patience = 2,
                                                lr_patience = 1,
                                                checkpoint_verbose=1,
                                                train_verbose=0,
                                                seed = 42)
    def test_compile(self):
        self.cache_setup()
        self.cache_compile()

        with open('tests/out/compile.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

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
            y_pred_json = json.dumps(np.argmax(self.y_pred,axis=1).astype(int),
                                     cls=NumpyArrayEncoder) 
            f.write("{  ")
            
            if write_tests:
                y_test_json = json.dumps(np.argmax(self.y_test,axis=1).astype(int), cls=NumpyArrayEncoder) 
                f.write(f"  'y_test': {y_test_json}\n")

            # print pred no matter what
            y_pred_json = json.dumps(np.argmax(self.y_pred,axis=1).astype(int), cls=NumpyArrayEncoder) 
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

        self.y_pred = self.model.predict(self.x_test) # xxx
        self.write_test_pred_pairs("compare_predictions.json")

        self.fcmp("compare_predictions.json")


    def test_evaluate_compare_pred(self):
        self.cache_setup()
        self.cache_compile()
        self.cache_fit()

        from sbr.evaluate import compare_predictions
        self.y_pred, pairs = compare_predictions(y_pred=self.model.predict(self.x_test),
                                                 y_test=self.y_test,
                                                 class_names=self.class_names,
                                                 verbose = True)

        self.write_test_pred_pairs("compare_predictions.json", pairs=pairs)
        self.fcmp("compare_predictions.json")
        


    # xxx do this next
    '''
    def test_evaluate_mislabeled(self):
        self.cache_setup()
        self.cache_compile()
        self.cache_fit()
        # xxx where to get label_df?
        mislabeled_counts, mislabeled = mislabeled_pair_counts(model=self.model, X=self.X, y=self.y, class_names=self.class_names,
                                                               sample_ids = pd.Series(label_df["sample_id"]),
                                                               batch_size=500)
        print(mislabeled_counts)
    '''
