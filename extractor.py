import radiomics 
import yaml
import os
import pandas as pd
from multiprocessing import Pool
from time import time
import sys 


class Extractor:
    def __init__(self, workers = 10, batch_size = 10):
        self.extract = None  
        self.workers = workers
        self.batch_size = batch_size
        self.columns = None 
        self.load_extractor()
        self.df = None


    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.dir_pairs = config["directories"]

    def get_column_names(self, features): 
        self.columns = ["seg_path"]
        for feature in features.keys():
            self.columns.append(feature)

    def get_scan_file(self, seg_path, scan_dir): 
        fn = os.path.basename(seg_path)
        fn = fn.replace(".nii.gz", "_0000.nii.gz")
        print(fn)
        scan_path = os.path.join(scan_dir, fn)
        return scan_path

    def get_pairs(self):
        self.pairs = []
        for scan_dir, seg_dir in self.dir_pairs: 
            segs = os.listdir(seg_dir)
            segs = [_ for _ in segs if _.endswith(".nii.gz")]
            for seg in segs: 
                seg_path = os.path.join(seg_dir, seg)
                scan_path = self.get_scan_file(seg_path, scan_dir)
                if os.path.exists(scan_path):
                    self.pairs.append((scan_path, seg_path))
                else:
                    print(f"Scan file {scan_path} does not exist.")

        print(f"Found {len(self.pairs)} pairs of scan and segmentation files.")
        return self.pairs

    def load_extractor(self):
        self.extract = radiomics.featureextractor.RadiomicsFeatureExtractor()
        self.extract.disableAllImageTypes()
        self.extract.enableAllFeatures()

    def enable_wavelet(self):  
        if not self.extract:
            self.load_extractor()
        
        self.extract.enableImageTypeByName("Wavelet")

    def enable_LoG(self): 
        param = {"sigma":[1, 2, 3, 4, 5]}
        if not self.extract:
            self.load_extractor()
        self.extract.enableImageTypeByName("LoG", customArgs=param)

    def worker(self, scan, seg):
        if not self.extract:
            self.load_extractor()
        features = self.extract.execute(scan, seg)
        print(features)
        return seg, features


    def parallel_feature_extraction(self, pairs):
        pairs = pairs[:2]
        num_pairs = len(pairs)

        for i in range(0, num_pairs, self.batch_size):
            right = min(num_pairs, i + self.batch_size)
            print(f"Processing batch {i} to {right}")
            with Pool(self.workers) as pool:
                batch_res = pool.starmap(self.worker, pairs[i:right])
            for seg, features in batch_res:
                if self.df is None:
                    self.get_column_names(features)
                    self.df = pd.DataFrame(columns=self.columns)
                row = [seg] + list(features.values())
                self.df.loc[len(self.df)] = row

        print(self.df) 
        return self.df


    


if __name__ == "__main__":
    ext = Extractor(workers=10, batch_size=10)
    # # Load config
    ext.load_config('data.yaml')
    ext.enable_LoG()
    ext.enable_wavelet()
    pairs = ext.get_pairs()
    # # Extract features
    df = ext.parallel_feature_extraction(ext.pairs)
    # # Save to CSV
    output_path = "features.csv"
    df.to_csv(output_path, index=False)
    print(df) 
