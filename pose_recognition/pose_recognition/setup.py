import setuptools

setuptools.setup(
    name="pose_recognition",
    version="1.0",
    author="David Olivier",
    author_email='david.olivier404@gmail.com',
    url='http://github.com/ClubSynapseETS/Prothese',
    description="Platform to build, train and deploy classifiers for movement recognition based on EMG data.",
    content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
)
