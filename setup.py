from distutils.core import setup

setup(
	name='FaceGen',
	version='1.0',
	packages=['experiments', 'models', 'utils'],
	author='Group 24',
	description='CS4245_Project',
	install_requires=['tqdm==4.64', 'numpy==1.22.3', 'torch==1.11.0', 'torchvision==0.12.0', 'scipy==1.8.1'],
	classifiers=[
		'Programming Language :: Python :: 3.9',
	],
)