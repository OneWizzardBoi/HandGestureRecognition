import setuptools

setuptools.setup(
    name="emg_comm",
    version="1.0",
    author="David Olivier",
    author_email='david.olivier404@gmail.com',
    url='http://github.com/ClubSynapseETS/Prothese',
    description="Allows the acquisition of data from the myo armband from Thalmic labs and potentialy other hardware.",
    content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
)
