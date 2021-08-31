import hashlib
import pickle
import os

class Cookie:
    
    def __init__(self, arr, folder="cookie"):
        s = " ".join(str(x) for x in arr)
        os.makedirs(folder, exist_ok=True)
        self.pickle_path = folder + "/" + hashlib.md5(s.encode('utf-8')).hexdigest() + ".cookie"
        self.folder = folder
        
    def read_cookie(self):
        if os.path.exists(self.pickle_path):
            try:
                print("Loading cookie:", self.pickle_path)
                with open(self.pickle_path, "rb") as f:
                    return pickle.load(f)
            except:
                print("Load cookie error!")
                pass

    def save_cookie(self, data):
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(data, f)
            print("Saved cookie:", self.pickle_path)
        return data
