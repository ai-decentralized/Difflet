import glob, os, sys, json
root = sys.argv[1]
names = set()
for j in sys.argv[2:]:
    d = json.load(open(j))
    for v in d["device_names"][0].values():
        names.add(v.split("neff_")[-1])
for f in glob.glob(root + "/**/*.neff", recursive=True):
    with open(f, "rb") as fh:
        h = fh.read(0x100)
    hid = h[0xac:0xbc].hex()
    mark = "TRACE" if hid in names else ""
    if os.path.getsize(f) > 300000 or mark:
        print(hid, os.path.getsize(f), f.replace(root, ""), mark)
