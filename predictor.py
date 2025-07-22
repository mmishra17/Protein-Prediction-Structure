import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

class ReverseProteinPredictor:
    def __init__(self):
        """Initialize the reverse protein structure predictor"""
        self.helix_propensities = {
            'A': 0.5, 'L': 0.5, 'M': 0.5, 'E': 0.4, 'K': 0.4,
            'R': 0.3, 'Q': 0.3, 'H': 0.3, 'D': 0.2, 'N': 0.2,
            'T': 0.2, 'S': 0.2, 'Y': 0.2, 'P': 0.1, 'G': 0.1,
            'V': 0.2, 'I': 0.2, 'F': 0.2, 'W': 0.2, 'C': 0.2
        }
        
        self.sheet_propensities = {
            'V': 0.5, 'I': 0.5, 'Y': 0.4, 'F': 0.4, 'T': 0.3,
            'W': 0.3, 'C': 0.3, 'L': 0.3, 'M': 0.2, 'A': 0.2,
            'R': 0.2, 'G': 0.1, 'D': 0.1, 'K': 0.2, 'S': 0.2,
            'H': 0.2, 'Q': 0.2, 'E': 0.2, 'P': 0.1, 'N': 0.1
        }
        
        self.coil_propensities = {
            'G': 0.5, 'P': 0.5, 'D': 0.4, 'N': 0.4, 'S': 0.3,
            'T': 0.3, 'R': 0.3, 'K': 0.3, 'Q': 0.3, 'E': 0.2,
            'H': 0.2, 'A': 0.2, 'M': 0.2, 'Y': 0.2, 'W': 0.2,
            'F': 0.1, 'L': 0.1, 'V': 0.1, 'I': 0.1, 'C': 0.1
        }
        
        self.svm_model = None
        self.label_encoder = LabelEncoder()
        self.initialize_svm()

    def initialize_svm(self):
        """Initialize and train the SVM model with known protein structure patterns"""
        X_train = []
        y_train = []
        
        for ss_type, propensities in [
            ('H', self.helix_propensities),
            ('E', self.sheet_propensities),
            ('C', self.coil_propensities)
        ]:
            for aa, prop in propensities.items():
                n_samples = int(prop * 100)
                features = self.get_amino_acid_features(aa)
                X_train.extend([features] * n_samples)
                y_train.extend([ss_type] * n_samples)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.svm_model = SVC(kernel='rbf', probability=True)
        self.svm_model.fit(X_train, y_train)

    def get_amino_acid_features(self, aa):

        return [
            self.helix_propensities.get(aa, 0),
            self.sheet_propensities.get(aa, 0),
            self.coil_propensities.get(aa, 0)
        ]

    def predict_sequence_svm(self, ss_sequence):
        sequence = []
        
        for ss in ss_sequence:
            if ss == 'H':
                propensities = self.helix_propensities
            elif ss == 'E':
                propensities = self.sheet_propensities
            else:  
                propensities = self.coil_propensities
            
            aa_probabilities = {}
            for aa in propensities.keys():
                features = self.get_amino_acid_features(aa)
                prob = self.svm_model.predict_proba([features])[0]
                ss_index = list(self.svm_model.classes_).index(ss)
                aa_probabilities[aa] = prob[ss_index] * propensities[aa]
            
            total_prob = sum(aa_probabilities.values())
            aa_probabilities = {k: v/total_prob for k, v in aa_probabilities.items()}
            
            amino_acids = list(aa_probabilities.keys())
            probabilities = list(aa_probabilities.values())
            sequence.append(np.random.choice(amino_acids, p=probabilities))
        
        return ''.join(sequence)

    def validate_ss_input(self, ss_sequence):
        ss_sequence = ss_sequence.upper()
        valid_ss = set('HEC')
        invalid_elements = set(ss_sequence) - valid_ss
        if invalid_elements:
            raise ValueError(f"Invalid secondary structure elements found: {invalid_elements}. Use only H (helix), E (sheet), or C (coil).")
        return ss_sequence

    def generate_3d_coordinates(self, ss_sequence):
        coords = []
        x, y, z = 0, 0, 0
        helix_rise = 1.5
        helix_radius = 2.3
        helix_angle = 100
        sheet_rise = 3.3
        sheet_spacing = 3.2

        for i, ss in enumerate(ss_sequence):
            if ss == 'H':
                angle = np.radians(i * helix_angle)
                x = helix_radius * np.cos(angle)
                y = helix_radius * np.sin(angle)
                z += helix_rise
            elif ss == 'E':
                x += sheet_spacing * np.cos(np.radians(45))
                y += sheet_spacing * np.sin(np.radians(45))
                z += (-1) ** (i // 2) * sheet_rise
            else:
                x += np.random.normal(0, 1.0)
                y += np.random.normal(0, 1.0)
                z += np.random.normal(0, 1.0)
            coords.append(np.array([x, y, z]))
        return np.array(coords)

    def calculate_ss_percentages(self, ss_sequence):
        total_length = len(ss_sequence)
        helix_percent = ss_sequence.count('H') / total_length * 100
        sheet_percent = ss_sequence.count('E') / total_length * 100
        coil_percent = ss_sequence.count('C') / total_length * 100
        return helix_percent, sheet_percent, coil_percent
    
    def visualize_structure(self, coords, ss_sequence, predicted_sequence):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        color_map = {'H': 'red', 'E': 'blue', 'C': 'green'}
        
        for i in range(len(coords) - 1):
            ax.plot(coords[i:i+2, 0], coords[i:i+2, 1], coords[i:i+2, 2],
                   color=color_map[ss_sequence[i]], linewidth=2,
                   label=f"{ss_sequence[i]}" if ss_sequence[i] not in ax.get_legend_handles_labels()[1] else "")

        for i, (coord, ss) in enumerate(zip(coords, ss_sequence)):
            ax.scatter(*coord, color=color_map[ss], s=100)
            if i % 5 == 0:
                ax.text(*coord, predicted_sequence[i], size=8)

        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('Predicted Protein Structure\nColored by Secondary Structure')
        
        handles = [plt.Line2D([0], [0], color=color, label=struct)
                for struct, color in color_map.items()]
        ax.legend(handles=handles, labels=['Helix', 'Sheet', 'Coil'])
        
        plt.show()

    def analyze_properties(self, sequence):
        analysis = ProteinAnalysis(sequence)
        return {
            'Molecular Weight (Da)': round(analysis.molecular_weight(), 2),
            'Aromaticity': round(analysis.aromaticity(), 3),
            'Instability Index': round(analysis.instability_index(), 2),
            'Isoelectric Point': round(analysis.isoelectric_point(), 2),
            'Secondary Structure Prediction Confidence': self.calculate_prediction_confidence(sequence)
        }

    def calculate_prediction_confidence(self, sequence):
        confidences = []
        for aa in sequence:
            features = self.get_amino_acid_features(aa)
            prob = self.svm_model.predict_proba([features])[0]
            confidences.append(max(prob))
        return round(np.mean(confidences) * 100, 2)

    def predict_structure(self, ss_sequence):
        try:
            ss_sequence = self.validate_ss_input(ss_sequence)
            predicted_sequence = self.predict_sequence_svm(ss_sequence) 
            coords = self.generate_3d_coordinates(ss_sequence)
            helix, sheet, coil = self.calculate_ss_percentages(ss_sequence)
            properties = self.analyze_properties(predicted_sequence)
            self.visualize_structure(coords, ss_sequence, predicted_sequence)
            
            return {
                'secondary_structure': ss_sequence,
                'predicted_sequence': predicted_sequence,
                'ss_percentages': {
                    'Helix': round(helix, 1),
                    'Sheet': round(sheet, 1),
                    'Coil': round(coil, 1)
                },
                'properties': properties
            }
        except Exception as e:
            raise Exception(f"Error in structure prediction: {e}")

if __name__ == "__main__":
    predictor = ReverseProteinPredictor()

    while True:
        ss_sequence = input("Enter secondary structure sequence (H=helix, E=sheet, C=coil) or 'quit' to exit: ")
        if ss_sequence.lower() == 'quit':
            break
        try:
            result = predictor.predict_structure(ss_sequence)
            print(f"Predicted Amino Acid Sequence: {result['predicted_sequence']}")
            print(f"Secondary Structure Percentages: {result['ss_percentages']}")
            print(f"Protein Properties: {result['properties']}")
        except Exception as e:
            print(f"Error: {e}")