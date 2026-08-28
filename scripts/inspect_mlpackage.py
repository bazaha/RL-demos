"""Read an .mlpackage's own spec, without coremltools.

Used by scripts/refresh_ios_model.sh as a build-time guard, so it has to be
dependency-free: it runs inside an Xcode build phase, where the `coreml` uv
group is not on PATH.

Checks, in order:
  1. every item Manifest.json advertises actually exists -- exactly the
     invariant coremlc reported as "Item does not exist for identifier: ..."
     when the app's .mlpackage was committed with only its Manifest.json
  2. the spec's real input feature name is the one the Swift side sends, and
     its outputs are the two the Swift side reads. This is the check that
     catches a rename made in the sources without re-exporting -- the normal
     case, since results/coreml_export/** is gitignored
  3. the spec's metadata mentions the checkpoint it should have come from, so a
     model exported from a different run is not staged silently

Prints exactly one line: "OK <detail>" or "FAIL <reason>".

The .mlmodel is a serialized CoreML.Specification.Model protobuf. Only three
field numbers are needed (Model.proto):
    Model.description        = 2   (ModelDescription; field 1 is specificationVersion)
    ModelDescription.input   = 1   (repeated FeatureDescription)
    ModelDescription.output  = 10  (repeated FeatureDescription)
    FeatureDescription.name  = 1   (string)
so a ~40-line wire-format walk beats taking a dependency. A plain string scan
does not work here: the input is named "x", and a single character matches
almost any binary by accident.
"""
import json
import os
import re
import sys


def _varint(buf, i):
    shift = result = 0
    while i < len(buf):
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7
        if shift > 63:
            break
    raise ValueError("truncated varint")


def fields(buf):
    """Yield (field_number, wire_type, payload) for one protobuf message."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _varint(buf, i)
            yield num, wire, val
        elif wire == 2:
            n, i = _varint(buf, i)
            if i + n > len(buf):
                raise ValueError("truncated length-delimited field")
            yield num, wire, buf[i:i + n]
            i += n
        elif wire == 5:
            yield num, wire, buf[i:i + 4]
            i += 4
        elif wire == 1:
            yield num, wire, buf[i:i + 8]
            i += 8
        else:
            raise ValueError("unsupported wire type %d" % wire)


def _feature_names(desc, field_number):
    names = []
    for num, wire, payload in fields(desc):
        if num != field_number or wire != 2:
            continue
        for n2, w2, p2 in fields(payload):
            if n2 == 1 and w2 == 2:
                names.append(p2.decode("utf-8", "replace"))
                break
    return names


def spec_interface(path):
    """(inputs, outputs) as declared by the model spec itself."""
    blob = open(path, "rb").read()
    for num, wire, payload in fields(blob):
        if num == 2 and wire == 2:          # Model.description (field 2; field 1 is specificationVersion)
            return (_feature_names(payload, 1), _feature_names(payload, 10)), blob
    raise ValueError("no ModelDescription in the spec")


def main(pkg, want_input, ckpt_tag):
    man = json.load(open(os.path.join(pkg, "Manifest.json")))
    bad = ["%s (%s)" % (k, v["path"])
           for k, v in man.get("itemInfoEntries", {}).items()
           if not os.path.exists(os.path.join(pkg, "Data", v["path"]))]
    if bad:
        return "FAIL Manifest.json advertises items that are not present: " + "; ".join(bad)

    spec = os.path.join(pkg, "Data", "com.apple.CoreML", "model.mlmodel")
    if not os.path.exists(spec):
        return "FAIL no model spec at %s" % spec
    try:
        (inputs, outputs), blob = spec_interface(spec)
    except Exception as exc:                            # noqa: BLE001
        return "FAIL could not parse %s (%s)" % (spec, exc)

    if inputs != [want_input]:
        return ("FAIL staged model takes input %s but the app sends %r -- "
                "every prediction would throw" % (inputs, want_input))
    for out in ("policy_logits", "value"):
        if out not in outputs:
            return "FAIL staged model outputs %s, missing %r" % (outputs, out)
    # user_defined_metadata is a protobuf map; a token scan is enough for it,
    # and the checkpoint name is long enough not to match by accident
    if ckpt_tag and ckpt_tag.encode() not in set(re.findall(rb"[ -~]{4,64}", blob)):
        return ("FAIL staged model's metadata does not mention %s -- it came "
                "from a different export; run scripts/refresh_ios_model.sh" % ckpt_tag)
    return "OK spec: %s -> %s; ckpt %s" % (inputs[0], ", ".join(outputs), ckpt_tag)


if __name__ == "__main__":
    print(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""))
