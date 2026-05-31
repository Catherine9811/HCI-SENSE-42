# HCI-SENSE-42: A multimodal Human-Computer Interaction dataset *from Simulated Environment* for Neurocognitive User State Evaluation

SENSE-42 is a publicly available, multimodal dataset designed to support the study of user state monitoring during extended computer interaction sessions via neurocognitive, physiological and behavioural responses. Combining high-resolution neurophysiological recordings with behavioral and subjective data, this dataset enables research on the alternations of attention, mental/physical fatigue, cognitive workload, and related subjective indices at a very early stage.

This repository contains the [source code of the experiment](https://github.com/Catherine9811/HCI-SENSE-42/tree/experiment) implemented by [PsychoPy v2024.2.3](https://www.psychopy.org/), as well as the [Python and R scripts](https://github.com/Catherine9811/HCI-SENSE-42/tree/master) to parse and analyze the HCI-SENSE-42 dataset.

The project includes both the raw dataset (available via [Zenodo](https://doi.org/10.5281/zenodo.20328098)) and the analysis codebase (available on [GitHub](https://github.com/Catherine9811/HCI-SENSE-42/), WIP) for replicability and reuse.

![Experiment Information](https://github.com/Catherine9811/HCI-SENSE-42/blob/master/assets/experiment_flow_new.jpg)


## Dataset Description

The dataset was collected from 42 participants over a 2-hour continuous interaction session, during which participants engaged in a series of designed tasks on a desktop computer with a mouse and keyboard. The experimental tasks were conducted within a fully simulated desktop operating system environment, designed to closely mimic real-world computer usage scenarios. This setup mirrors how people typically use computers in daily life. The simulated experiment program also enables the comprehensive capture of the mouse and keyboard data, synchronized with a high refresh rate monitor at 144 Hz. Recordings were collected in a noise-insulated room, minimizing external interruptions from the environment or the experimenter.

The dataset includes multimodal data, including

- Behavioural Data
- 32-channel Electroencephalogram Recordings with BioSemi ActiveTwo System
- Respiratory Cycles with Respiration Belt
- 3-lead Electrocardiogram Recordings*
- Webcam Recordings*

[*] published data modality is limited to participants consented for sharing them to the public

## Repository Structure
```bash
├── README.md  # This README file
├── LICENSE    # CC0 License file
├── data    # Location for downloaded dataset files
│   ├── 001_explorer_2025-02-15_15h23.13.921.csv
│   ├── 001_explorer_2025-02-15_15h23.13.921.psydat
│   ├── ...  # More behavioural data files expected here
│   ├── ECG  # ECG recording files (*.fif)
│   │   ├── P002.fif
│   │   ├── ...
│   ├── EEG  # EEG recording files (*.bdf)
│   │   ├── P001.bdf
│   │   ├── ...
│   ├── Respiration  # Respiratory files (*.wav)
│   │   ├── P001.wav
│   │   ├── ...
│   ├── Events.txt   # Definition of EEG events
│   └── participant_enrollment.csv  # Encoded participant information
├── analyze_*  # Contains Python scripts for data analysis
│   ├── check_*.py    # Basic parsing and visualization script
│   ├── convert_*.py  # Data parsing and format conversion script
│   ├── ...
├── condition  # Analyze the data with the condition given
│   ├── with_comfort
│   ├── with_handedness
│   ├── with_keyboard_frequency
│   ├── with_os
│   └── with_usage_hours
├── correlation  # Analyze the dataset by investigating the correlation
│   └── with_questionnaire
├── data_definition.py  # Definition of the list of *.psydat files
├── data_parser.py      # Definition of the *.psydat parsing class
├── main.py             # A small example script to parse for sleepiness levels across time domain
├── requirements.txt    # Python package requirement file
└── test_materials      # Analysis related to typing materials
    └── check_text_materials.py  # Parsing and visualization of the word distributions
```

> 💡 After downloaded the [HCI-SENSE-42 Dataset](https://doi.org/10.5281/zenodo.20328098), we expect them to be organized in the format listed above.

## Getting Started

Prerequisites
- Python 3.9+ installed
- pip available
- RStudio installed (to run R scripts)

### Clone this repository
```bash
git clone https://github.com/Catherine9811/HCI-SENSE-42.git
cd HCI-SENSE-42
```

### Install the dependencies
```bash
pip install -r requirements.txt
```
> 💡 The `numpy` package version must match the `PsychoPy` package versions. In my case, numpy==1.26.4 and psychopy==2024.2.3

### Run a quick test
```bash
python3 main.py
```

### Accessing the Dataset

The dataset is hosted on [Zenodo](https://doi.org/10.5281/zenodo.20328098): https://doi.org/10.5281/zenodo.20328098 and partially hosted on [Synapse](https://www.synapse.org/HCI_SENSE_42).

Data from sensors with different modalities are flattened to allow for separete downloads if not all of them are required in the analysis.

> 💡 You must login and agree to the conditions for use before accessing the webcam recordings as it contains sensitive information.

## Data Preprocessing
### 3-lead Electrocardiogram Recordings
![ECG Electrodes Placement Information](https://github.com/Catherine9811/HCI-SENSE-42/blob/master/assets/ECG_placement_convention.jpg)

ECG signals are embedded in the external channels of the RAW EEG recording files and the electrode placement conventions is shown in the image above.


### Respiratory Cycles

Respiratory cycles are recorded in the `Resp` channel of the RAW EEG recording files.

We used [RespInPeace](https://github.com/mwlodarczak/RespInPeace) to process and analyze breathing belt data and saved them as `.wav` files in 32 Hz.

![Detected Keypoints from RespInPeace for P001](https://github.com/Catherine9811/HCI-SENSE-42/blob/master/assets/RespInPeace_output.png)

### 32-channel Electroencephalogram Recordings

It is recommended by BioSemi to apply `average` on all the electrodes before other processing steps to enhance the effective signal-to-noise ratio.

## Citation
If you find our work useful, please cite:
```bibtex
@misc{
  HCI-SENSE-42,
  title={SENSE-42 A multimodal dataset from a Simulated Environment for Neurocognitive State Evaluation during Human-Computer Interaction},
  url={https://repo-prod.prod.sagebase.org/repo/v1/doi/locate?portalId=1&id=syn68713182&type=ENTITY},
  DOI={10.7303/SYN68713182},
  publisher={Synapse},
  author={Zhang, Sai and Bai, Xinyu and Noreika, Valdas}, year={2025}
}
```
