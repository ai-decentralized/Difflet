"""Extract a NEFF (by its header id) from an NxD model.pt (torchscript zip)."""
import struct
import sys
import zipfile

pt, hexid, out = sys.argv[1], sys.argv[2], sys.argv[3]
z = zipfile.ZipFile(pt)
data = z.read("model/data.pkl")
pat = bytes.fromhex(hexid)
idx = data.find(pat)
while idx >= 0:
    start = idx - 0xAC
    hdr = data[start:start + 0x400]
    hsize = struct.unpack("<Q", hdr[8:16])[0]
    psize = struct.unpack("<Q", hdr[16:24])[0]
    print("found at", idx, "start", start, "header", hsize, "payload", psize)
    if hsize == 0x400:
        with open(out, "wb") as fh:
            fh.write(data[start:start + hsize + psize])
        print("wrote", out, hsize + psize)
        break
    idx = data.find(pat, idx + 1)
