# SENSE-42: A multimodal dataset from a Simulated Environment for Neurocognitive State Evaluation during Human-Computer Interaction

This repository contains the [source code of the experiment](https://github.com/Catherine9811/SENSE-42-HCI/tree/experiment) implemented by [PsychoPy v2024.2.3](https://www.psychopy.org/), as well as the [Python and R scripts](https://github.com/Catherine9811/SENSE-42-HCI/tree/master) to parse and analyze the SENSE-42-HCI dataset.

The dataset is publicly available on [Synapse](https://www.synapse.org/SENSE_42_HCI): https://www.synapse.org/SENSE_42_HCI

![Experiment Information](https://github.com/Catherine9811/SENSE-42-HCI/blob/master/assets/experiment_flow.jpg)


## Dataset Description

The SENSE-42-HCI dataset includes multimodal data captured in a 2-hour session for 42 participants, including
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

> 💡 After downloaded the [SENSE-42-HCI Dataset](https://www.synapse.org/Synapse:syn68714673), we expect them to be organized in the format listed above.

## Getting Started

Prerequisites
- Python 3.9+ installed
- pip available
- RStudio installed (to run R scripts)

### Clone this repository
```bash
git clone https://github.com/Catherine9811/SENSE-42-HCI.git
cd SENSE-42-HCI
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

The dataset is hosted on [Synapse](https://www.synapse.org/SENSE_42_HCI): https://www.synapse.org/SENSE_42_HCI

Data from sensors with different modalities are flattened to allow for separete downloads if not all of them are required in the analysis.

> 💡 You must login and agree to the conditions for use before accessing the webcam recordings as it contains sensitive information.

## Data Preprocessing
### 3-lead Electrocardiogram Recordings
![ECG Electrodes Placement Information](https://github.com/Catherine9811/SENSE-42-HCI/blob/master/assets/ECG_placement_convention.jpg)

ECG signals are embedded in the external channels of the RAW EEG recording files and the electrode placement conventions is shown in the image above.


### Respiratory Cycles

Respiratory cycles are recorded in the `Resp` channel of the RAW EEG recording files.

We used [RespInPeace](https://github.com/mwlodarczak/RespInPeace) to process and analyze breathing belt data and saved them as `.wav` files in 32 Hz.

![Detected Keypoints from RespInPeace for P001](https://github.com/Catherine9811/SENSE-42-HCI/blob/master/assets/RespInPeace_output.png)

### 32-channel Electroencephalogram Recordings

It is recommended by BioSemi to apply `average` on all the electrodes before other processing steps to enhance the effective signal-to-noise ratio.

## Citation
If you find our work useful, please cite:
```bibtex
@misc{
  SENSE-42-HCI,
  title={SENSE-42 A multimodal dataset from a Simulated Environment for Neurocognitive State Evaluation during Human-Computer Interaction},
  url={https://repo-prod.prod.sagebase.org/repo/v1/doi/locate?portalId=1&id=syn68713182&type=ENTITY},
  DOI={10.7303/SYN68713182},
  publisher={Synapse},
  author={Zhang, Sai and Bai, Xinyu and Noreika, Valdas}, year={2025}
}
```
