# ReferenceRunRank

The code in this repository assumes the use of code in the ML4DQMDC-PixealAE to access OMS.

In the future we may put all data augmentation utilities in a common repository, but for the first implementation we'll rely on cloning the other repository and use the OMS utility functions from there. 

As mentioned in the README.md file of the ML4DQMDC-PixealAE repository (in the OMSAPI directory https://github.com/CMSTrackerDPG/ML4DQMDC-PixelAE/blob/d8a8dffd2105303ddbfd243f19c8388492baaf26/omsapi/README.md), the user will need to setup the OMSAPI, by creating a clientid.py file with their authentication information. 
Due to recent changes in the authentication, you may need to update the OMS libraries: 
```
#Update
cd oms-api-client
git pull
python3 setup.py install --user
#Build for CC7:
python3.6 setup.py bdist_rpm --python python3.6 --release 0.cc7
#Build for CC8:
python3.8 setup.py bdist_rpm --python python3.8 --release 0.cc8
```
