# Data Access and Restricted-Data Handling

This repository is designed for use with the EchoNext dataset under the applicable PhysioNet/EchoNext access terms. The repository does not distribute the dataset.

## What May Be Public

The following are appropriate for public release in the GitHub repository:
- source code
- configuration files
- notebooks
- documentation
- aggregate summary tables
- aggregate benchmark figures
- manuscript-related text and reproducibility instructions

## What Should Remain Private

The following should not be committed to a public repository or shared publicly unless explicitly permitted by the governing data use agreement:
- raw dataset files
- copied dataset folders in cloud storage
- patient-level prediction CSVs
- row-level outputs containing `patient_key`, `row_index`, or similar identifiers
- unrestricted sharing links to private Google Drive dataset folders

## Recommended Workflow

- Each collaborator should obtain their own approved access to the EchoNext dataset.
- Code should be shared through GitHub.
- Data should be stored privately, for example in a private Google Drive folder or secure local storage.
- Colab notebooks should clone code from GitHub and read data from a private Drive folder owned by the authorized user.
- Public artifacts should be limited to code and aggregate results.

## Security Notes

- Do not make the dataset folder publicly accessible in Google Drive.
- Do not distribute downloaded copies of the restricted dataset to collaborators who do not have their own approved access.
- Treat patient-level derived outputs as restricted artifacts.
- If potentially identifying information is discovered in the dataset, follow the reporting instructions in the applicable data use agreement.

## Responsibility

Users of this repository are responsible for ensuring that their use of the code and the underlying dataset complies with the applicable data use agreement, institutional policies, and journal requirements.
