# Installation on top of QSFNet

The public RACFNet package deliberately does not redistribute the QSFNet
backbone because the upstream repository currently has no explicit software
license. Obtain the upstream source directly from its authors and apply the
RACFNet-original overlay:

```bash
git clone https://github.com/Lx-Bao/QSFNet.git
git clone https://github.com/hylandelisandro-droid/RACFNet.git RACFNet
python RACFNet/tools/install_into_qsfnet.py --qsfnet QSFNet --racfnet RACFNet
cd QSFNet
python -m pip install -r requirements.racfnet.txt
```

The installer performs these bounded changes:

1. copies the RACFNet-original modules, experiment drivers, evaluators and
   configurations into the QSFNet checkout;
2. copies upstream `Tnet.py` to `round2/RACFNet.py` locally;
3. checks and applies `patches/qsfnet_to_racfnet.patch` to that local copy;
4. leaves the original upstream files unchanged.

If the upstream implementation changes and `git apply --check` fails, use the
QSFNet revision compatible with the patch rather than forcing the patch. The
public package does not claim ownership of the upstream source.
