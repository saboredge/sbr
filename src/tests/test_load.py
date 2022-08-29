#!/usr/bin/env python
import sys
sys.path.insert(0, f'.')

import numpy as np

from unittest import TestCase

class TestSBR(TestCase):
    load_done = False
    setup_done = False

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

        

def main():
    t = TestSBR()
    t.test_load()
    t.test_setup_classnames()
    t.test_setup_first_label()

if __name__ == "__main__":
    main()

