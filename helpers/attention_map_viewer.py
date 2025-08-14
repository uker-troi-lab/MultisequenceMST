import os
import sys
import csv
import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QLabel, QListWidget, QRadioButton, QPushButton, QButtonGroup, 
                              QGroupBox, QSplitter, QStatusBar, QScrollArea, QFrame)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QSize

class AttentionMapViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Attention Map Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Path to attention maps
        self.attention_maps_path = r"/path/to/attention/maps"
        
        # Variables to store ratings
        self.current_case = None
        self.attention_rating = None
        self.slice_rating = None
        
        # Results storage
        self.results = {}
        self.csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attention_ratings.csv")
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Create left panel for case list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create case list
        list_label = QLabel("Case List:")
        left_layout.addWidget(list_label)
        
        self.case_listbox = QListWidget()
        self.case_listbox.setMinimumWidth(250)
        self.case_listbox.currentItemChanged.connect(self.on_case_select)
        left_layout.addWidget(self.case_listbox)
        
        # Add left panel to splitter
        splitter.addWidget(left_panel)
        
        # Create right panel for images and ratings
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create image display area
        image_scroll_area = QScrollArea()
        image_scroll_area.setWidgetResizable(True)
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        
        # Create top row with overlay and overlay slice images
        top_row = QHBoxLayout()
        
        # Overlay image
        overlay_group = QGroupBox("Overlay Image")
        overlay_layout = QVBoxLayout(overlay_group)
        self.overlay_label = QLabel()
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setMinimumSize(400, 400)
        self.overlay_label.setFrameShape(QFrame.Box)
        overlay_layout.addWidget(self.overlay_label)
        top_row.addWidget(overlay_group)
        
        # Overlay slice image
        slice_group = QGroupBox("Overlay Slice Image")
        slice_layout = QVBoxLayout(slice_group)
        self.slice_label = QLabel()
        self.slice_label.setAlignment(Qt.AlignCenter)
        self.slice_label.setMinimumSize(400, 400)
        self.slice_label.setFrameShape(QFrame.Box)
        slice_layout.addWidget(self.slice_label)
        top_row.addWidget(slice_group)
        
        image_layout.addLayout(top_row)
        
        # Input image (large, at the bottom)
        input_group = QGroupBox("Input Image")
        input_layout = QVBoxLayout(input_group)
        self.input_label = QLabel()
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setMinimumSize(800, 400)
        self.input_label.setFrameShape(QFrame.Box)
        input_layout.addWidget(self.input_label)
        image_layout.addWidget(input_group)
        
        image_scroll_area.setWidget(image_widget)
        right_layout.addWidget(image_scroll_area)
        
        # Create ratings area
        ratings_layout = QHBoxLayout()
        
        # Attention map rating
        attention_group = QGroupBox("Attention Map Rating")
        attention_layout = QVBoxLayout(attention_group)
        
        self.attention_group = QButtonGroup(self)
        self.good_attention = QRadioButton("Good")
        self.moderate_attention = QRadioButton("Moderate")
        self.bad_attention = QRadioButton("Bad")
        
        self.attention_group.addButton(self.good_attention, 1)
        self.attention_group.addButton(self.moderate_attention, 2)
        self.attention_group.addButton(self.bad_attention, 3)
        
        attention_layout.addWidget(self.good_attention)
        attention_layout.addWidget(self.moderate_attention)
        attention_layout.addWidget(self.bad_attention)
        
        ratings_layout.addWidget(attention_group)
        
        # Slice rating
        slice_rating_group = QGroupBox("Slice Rating")
        slice_rating_layout = QVBoxLayout(slice_rating_group)
        
        self.slice_group = QButtonGroup(self)
        self.good_slice = QRadioButton("Good")
        self.moderate_slice = QRadioButton("Moderate")
        self.bad_slice = QRadioButton("Bad")
        
        self.slice_group.addButton(self.good_slice, 1)
        self.slice_group.addButton(self.moderate_slice, 2)
        self.slice_group.addButton(self.bad_slice, 3)
        
        slice_rating_layout.addWidget(self.good_slice)
        slice_rating_layout.addWidget(self.moderate_slice)
        slice_rating_layout.addWidget(self.bad_slice)
        
        ratings_layout.addWidget(slice_rating_group)
        
        right_layout.addLayout(ratings_layout)
        
        # Create buttons layout
        buttons_layout = QHBoxLayout()
        
        # Create save button
        self.save_button = QPushButton("Save Ratings")
        self.save_button.clicked.connect(self.save_ratings)
        self.save_button.setMinimumHeight(40)
        buttons_layout.addWidget(self.save_button)
        
        # Create load button
        self.load_button = QPushButton("Load Ratings from CSV")
        self.load_button.clicked.connect(self.load_ratings_from_csv)
        self.load_button.setMinimumHeight(40)
        buttons_layout.addWidget(self.load_button)
        
        right_layout.addLayout(buttons_layout)
        
        # Add right panel to splitter
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([300, 900])
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Load existing results and populate case list
        self.load_existing_results()
        self.populate_case_list()
        
        # Initialize with empty images
        self.clear_images()
    
    def populate_case_list(self):
        # Get all case directories
        try:
            cases = sorted(os.listdir(self.attention_maps_path))
            for case in cases:
                self.case_listbox.addItem(case)
                
                # If this case has ratings, set its text color to blue
                if case in self.results:
                    item = self.case_listbox.findItems(case, Qt.MatchExactly)[0]
                    item.setForeground(Qt.blue)
            
            self.status_bar.showMessage(f"Loaded {len(cases)} cases")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading cases: {str(e)}")
    
    def on_case_select(self, current, previous):
        if not current:
            return
        
        # Save the current selection before switching to a new case
        if previous and self.current_case:
            attention_rating = self.get_attention_rating()
            slice_rating = self.get_slice_rating()
            
            # Save even if only one rating is selected
            if attention_rating or slice_rating:
                # Get existing ratings if available
                existing_data = self.results.get(self.current_case, {})
                
                # Update with new ratings, preserving any existing rating that wasn't changed
                if attention_rating:
                    existing_data['attention_rating'] = attention_rating
                if slice_rating:
                    existing_data['slice_rating'] = slice_rating
                
                self.results[self.current_case] = existing_data
                
                # Mark case as rated in the listbox
                item = self.case_listbox.findItems(self.current_case, Qt.MatchExactly)[0]
                item.setForeground(Qt.blue)
                
                # Save to CSV
                self.save_to_csv()
        
        case = current.text()
        self.current_case = case
        
        # Load images
        self.load_case_images(case)
        
        # Load existing ratings if available
        if case in self.results:
            rating_data = self.results[case]
            
            # Set attention map rating
            if rating_data['attention_rating'] == 'good':
                self.good_attention.setChecked(True)
            elif rating_data['attention_rating'] == 'moderate':
                self.moderate_attention.setChecked(True)
            elif rating_data['attention_rating'] == 'bad':
                self.bad_attention.setChecked(True)
            
            # Set slice rating
            if rating_data['slice_rating'] == 'good':
                self.good_slice.setChecked(True)
            elif rating_data['slice_rating'] == 'moderate':
                self.moderate_slice.setChecked(True)
            elif rating_data['slice_rating'] == 'bad':
                self.bad_slice.setChecked(True)
        else:
            # Clear ratings
            self.attention_group.setExclusive(False)
            self.good_attention.setChecked(False)
            self.moderate_attention.setChecked(False)
            self.bad_attention.setChecked(False)
            self.attention_group.setExclusive(True)
            
            self.slice_group.setExclusive(False)
            self.good_slice.setChecked(False)
            self.moderate_slice.setChecked(False)
            self.bad_slice.setChecked(False)
            self.slice_group.setExclusive(True)
        
        self.status_bar.showMessage(f"Loaded case: {case}")
    
    def load_case_images(self, case):
        try:
            case_path = os.path.join(self.attention_maps_path, case)
            
            # Get image file names
            input_file = f"input_{case.split('_', 1)[1]}.png"
            overlay_file = f"overlay_{case.split('_', 1)[1]}.png"
            slice_file = f"overlay_{case.split('_', 1)[1]}_slice.png"
            
            # Load images
            input_path = os.path.join(case_path, input_file)
            overlay_path = os.path.join(case_path, overlay_file)
            slice_path = os.path.join(case_path, slice_file)
            
            # Check if files exist
            if not os.path.exists(input_path):
                self.status_bar.showMessage(f"Error: Input image not found: {input_file}")
                return
            
            if not os.path.exists(overlay_path):
                self.status_bar.showMessage(f"Error: Overlay image not found: {overlay_file}")
                return
            
            if not os.path.exists(slice_path):
                self.status_bar.showMessage(f"Error: Slice image not found: {slice_file}")
                return
            
            # Load images
            self.input_label.setPixmap(QPixmap(input_path).scaled(
                800, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            self.overlay_label.setPixmap(QPixmap(overlay_path).scaled(
                400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            self.slice_label.setPixmap(QPixmap(slice_path).scaled(
                400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
        except Exception as e:
            self.status_bar.showMessage(f"Error loading images: {str(e)}")
            self.clear_images()
    
    def clear_images(self):
        # Create empty pixmaps
        empty_pixmap = QPixmap(400, 400)
        empty_pixmap.fill(Qt.white)
        
        empty_large_pixmap = QPixmap(800, 400)
        empty_large_pixmap.fill(Qt.white)
        
        # Update labels
        self.input_label.setPixmap(empty_large_pixmap)
        self.overlay_label.setPixmap(empty_pixmap)
        self.slice_label.setPixmap(empty_pixmap)
    
    def get_attention_rating(self):
        if self.good_attention.isChecked():
            return "good"
        elif self.moderate_attention.isChecked():
            return "moderate"
        elif self.bad_attention.isChecked():
            return "bad"
        return None
    
    def get_slice_rating(self):
        if self.good_slice.isChecked():
            return "good"
        elif self.moderate_slice.isChecked():
            return "moderate"
        elif self.bad_slice.isChecked():
            return "bad"
        return None
    
    def save_ratings(self):
        if not self.current_case:
            self.status_bar.showMessage("No case selected")
            return
        
        attention_rating = self.get_attention_rating()
        slice_rating = self.get_slice_rating()
        
        if not attention_rating or not slice_rating:
            self.status_bar.showMessage("Please rate both attention map and slice")
            return
        
        # Save ratings
        self.results[self.current_case] = {
            'attention_rating': attention_rating,
            'slice_rating': slice_rating
        }
        
        # Mark case as rated in the listbox
        item = self.case_listbox.findItems(self.current_case, Qt.MatchExactly)[0]
        item.setForeground(Qt.blue)
        
        # Save to CSV
        self.save_to_csv()
        
        self.status_bar.showMessage(f"Saved ratings for case: {self.current_case}")
    
    def save_to_csv(self):
        try:
            with open(self.csv_path, 'w', newline='') as csvfile:
                fieldnames = ['case', 'attention_rating', 'slice_rating']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for case, data in self.results.items():
                    writer.writerow({
                        'case': case,
                        'attention_rating': data['attention_rating'],
                        'slice_rating': data['slice_rating']
                    })
            
            self.status_bar.showMessage(f"Saved ratings to {self.csv_path}")
        except Exception as e:
            self.status_bar.showMessage(f"Error saving CSV: {str(e)}")
    
    def load_existing_results(self):
        self.load_results_from_file(self.csv_path)
    
    def load_results_from_file(self, file_path):
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        self.results[row['case']] = {
                            'attention_rating': row['attention_rating'],
                            'slice_rating': row['slice_rating']
                        }
                
                self.status_bar.showMessage(f"Loaded {len(self.results)} existing ratings")
                
                # Update case list to show rated cases
                for case in self.results:
                    items = self.case_listbox.findItems(case, Qt.MatchExactly)
                    if items:
                        items[0].setForeground(Qt.blue)
                        
            except Exception as e:
                self.status_bar.showMessage(f"Error loading ratings: {str(e)}")
    
    def load_ratings_from_csv(self):
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Ratings CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.load_results_from_file(file_path)

def main():
    app = QApplication(sys.argv)
    window = AttentionMapViewer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
