import json

def check(part):
    with open(f"results/stage7c_partitions/{part}_distribution.json", "r") as f:
        d = json.load(f)
    for k, v in d.items():
        n = v["num_samples"]
        if n % 64 == 1:
            print(f"{part} Client {k} has {n} samples -> batch remainder is 1!")
            return False
        if n < 64:
            print(f"{part} Client {k} has < 64 samples!")
            return False
    print(f"{part} is BatchNorm safe.")
    return True

check("d1")
check("d2")
