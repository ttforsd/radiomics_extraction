import radiomics 
import yaml
import os
import pandas as pd
from multiprocessing import Pool
import time 
import sys 
from collections import defaultdict 
import SimpleITK as sitk
import numpy as np
# from numpy import array



class Extractor:
    def __init__(self, workers = 3, batch_size = 1000, geometryTolerance=1):
        self.geometryTolerance = geometryTolerance
        self.extract = None  
        self.workers = workers
        self.batch_size = batch_size
        self.columns = None 
        self.load_extractor()
        self.df = None
        self.output_dir = "./results"

        self.load_config("data.yaml")
        # self.enable_LoG()
        # self.enable_wavelet()
        # self.enable_other_img_types()
        self.get_pairs()
        self.zero_ref = None


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
        scan_path = os.path.join(scan_dir, fn)
        return scan_path

    def get_pairs(self):
        self.pairs = defaultdict(list)
        for name, scan_dir, seg_dir in self.dir_pairs: 
            segs = os.listdir(seg_dir)
            segs = [_ for _ in segs if _.endswith(".nii.gz")]
            for seg in segs: 
                seg_path = os.path.join(seg_dir, seg)
                scan_path = self.get_scan_file(seg_path, scan_dir)
                if os.path.exists(scan_path):
                    self.pairs[name].append((scan_path, seg_path))
                else:
                    print(f"Scan file {scan_path} does not exist.")

            print(f"Found {len(self.pairs[name])} pairs of scan and segmentation files. with name {name}")
        return self.pairs

    def load_extractor(self):
        self.extract = radiomics.featureextractor.RadiomicsFeatureExtractor()
        self.extract.loadParams("param.yaml")
        # self.extract.disableAllImageTypes()
        self.extract.enableAllFeatures()
        # self.extract.settings = {
        #     'geometryTolerance': self.geometryTolerance,}
        # self.extract.settings["resamplePixelSpacing"] = [1, 1, 2]
        print(self.extract.settings)

    def enable_wavelet(self):  
        if not self.extract:
            self.load_extractor()
        
        self.extract.enableImageTypeByName("Wavelet")

    def enable_LoG(self): 
        param = {"sigma":[1, 2, 3, 4, 5]}
        if not self.extract:
            self.load_extractor()
        self.extract.enableImageTypeByName("LoG", customArgs=param)

    def enable_other_img_types(self): 
        if not self.extract:
            self.load_extractor()
        self.extract.enableImageTypeByName("Square")
        self.extract.enableImageTypeByName("SquareRoot")
        self.extract.enableImageTypeByName("Logarithm")
        self.extract.enableImageTypeByName("Exponential")
        self.extract.enableImageTypeByName("Gradient")
        self.extract.enableImageTypeByName("LBP2D")
        self.extract.enableImageTypeByName("LBP3D")



        

    def worker(self, scan, seg):
        # print time taken to process each pair
        start = time.time() 
        print(f"Extracting feactures from {scan} and {seg}")
        if not self.extract:
            self.load_extractor()
        try: 
            features = self.extract.execute(scan, seg)
        except ValueError as e:
            print(f"ValueError: {e}")
            if self.zero_ref is None:
                print("Zero reference not found. Getting zero reference.")
                self.zero_ref = self.get_zero_features()
            features = self.zero_ref
        end = time.time() 
        current_time = time.strftime('%l:%M%p %z on %b %d, %Y')
        print(f"{current_time} | Processed {scan} and {seg} in {end - start} seconds")

        return seg, features

    def get_zero_features(self): 
        _, features = self.worker("ref.nii.gz", "ref.nii.gz")
        for key in features: 
            features[key] = 0 
        return features


    def parse_output(self, seg, features):
        # if any feature len 1 np array, convert to float
        for feature in features.keys():
            if isinstance(features[feature], np.ndarray) and features[feature].size == 1:
                features[feature] = float(features[feature])
            if feature.startswith("diagnostic"):
                features[feature] = str(features[feature])
        
        return [seg] + list(features.values())



    def parallel_feature_extraction(self, pairs):
        res_arr = [] 
        pairs = pairs
        num_pairs = len(pairs)
        with Pool(self.workers) as pool:
            for i in range(0, num_pairs, self.batch_size):
                right = min(num_pairs, i + self.batch_size)
                print(f"Processing batch {i} to {right}")
                batch_res = pool.starmap(self.worker, pairs[i:right])
                for seg, features in batch_res:
                    res_arr.append(self.parse_output(seg, features))
        if self.df is None:
            self.get_column_names(features)
        self.df = pd.DataFrame(res_arr, columns=self.columns)

        print(self.df) 
        return self.df
    
    def save_data(self, name):
        if self.df is not None:
            output_path = os.path.join(self.output_dir, name)
            self.df.to_csv(f"{output_path}.csv", index=False)
            # save to parquet
            self.df.to_parquet(f"{output_path}.parquet", index=False)
            print(f"Data saved to {output_path}")
            self.df = None
        else:
            print("No data to save.")

    def check_match(self, seg_dir, scan_dir, geometryTolerance=0.01):
        segs = os.listdir(seg_dir)
        segs = [_ for _ in segs if _.endswith(".nii.gz")]
        invalid = [] 
        for seg in segs: 
            scan = seg.replace(".nii.gz", "_0000.nii.gz")
            scan_path = os.path.join(scan_dir, scan)
            if not os.path.exists(scan_path):
                print(f"Scan file {scan_path} does not exist.")
                continue 
            seg_path = os.path.join(seg_dir, seg)
            print(f"Checking {seg_path} and {scan_path}")
            scan, seg = sitk.ReadImage(scan_path), sitk.ReadImage(seg_path)
            try: 
                radiomics.imageoperations.checkMask(scan, seg, geometryTolerance=geometryTolerance)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Segmentation {seg_path} does not match scan {scan_path}")
                invalid.append(seg_path)
        return invalid
                



    def main(self): 
        for name in self.pairs: 
            # check if name.csv and name.parquet already exists, skip if they do
            output_path = os.path.join(self.output_dir, name)
            if os.path.exists(f"{output_path}.csv") and os.path.exists(f"{output_path}.parquet"):
                print(f"Data for {name} already exists. Skipping.")
                continue
            else: 
                print(f"Feature extraction for {name} starting.")
            pairs = self.pairs[name]
            self.parallel_feature_extraction(pairs)
            self.save_data(name)




    


if __name__ == "__main__":
    # get argv 
    num_workers = sys.argv[1]
    batch_size = sys.argv[2]
    if num_workers is not None:
        num_workers = int(num_workers)
    if batch_size is not None:
        batch_size = int(batch_size)
    ext = Extractor(workers=num_workers, batch_size=batch_size)
    ext.main()