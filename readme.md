# Deriva

Deriva is a predictive modeling project designed to forecast the dispersion and path of oil spills in the ocean. By simulating key environmental factors like ocean currents, wind, and oil properties, Deriva provides crucial, time-sensitive forecasts to help guide cleanup operations, protect vulnerable marine ecosystems, and minimize the devastating impact of spills.

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

```bash
git clone https://github.com/yourusername/deriva.git
cd deriva
conda env create --file env/conda.yaml # or mamba
conda activate deriva
pip install -r env/requirements.txt
```

## Datasets

To save disk space, the data converted from trajectories to tensors is transformed into integers with the following scales applied:
Current and Wind Data: x100

## Usage

```bash
# Example usage command
python simulate.py
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the [Your License Here].

## Contact

For questions or support, contact [your.email@example.com].
