import os
import sys
import time
import librosa
import tempfile
import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QIcon
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer, QUrl
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.fft import rfft, rfftfreq, irfft
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy import signal


class Signal:
    def __init__(self, file_path):
        self.file_path = file_path


class Equalizer(Signal):
    def __init__(self, ui, file_path):
        super().__init__(file_path)
        self.ui = ui
        self.timer = QTimer()
        self.media_player = QMediaPlayer()  # Create QMediaPlayer for audio playback
        self.media_player.setVolume(50)  # Adjust volume as needed
        self.media = self.media_player.duration()
        self.sample_rate = 0  # WAV file sample_rate
        self.data = []  # WAV file data values
        self.fft_data = []  # data after fft
        self.new_fft_data = []  # data after slicing and windowing
        self.time = []  # WAV file time x-axis
        self.index = 0  # Position for update
        self.duration = 0  # time of the input signal in seconds from librosa
        self.frequencies = []  # fft x-axis
        self.sliced_data = {}
        self.slider_values = {}
        self.modified_data = {}  # after windowing
        self.window_length = None

        ########################################### UI Objects ###########################################
        ##################################################################################################
        self.input_signal_graph = None
        self.fft_graph = None
        self.output_graph = None
        self.radio_btn = None
        self.figure = None
        self.canvas = None
        self.mode = {}
        self.slider_name = None
        self.selected_mode = None
        self.output_figure = None
        self.play_pause_btn = None
        self.speed_value = None
        self.audio_volume_slider = None
        self.audio_mute_btn = None
        self.output_canvas = None
        self.hide_input_spectro_checkbox = None
        self.hide_output_spectro_checkbox = None
        self.zoomin_btn = None
        self.zoomout_btn = None
        self.reset_btn = None
        self.std_doubleSpinBox = None
        self.std__label = None

        # estimated ranges for each animal and instrument
        self.animal_freq_ranges = {
            "dog": (2000, 4000),  # Monkey
            "cat": (800, 2000),  # Frog
            "whale": (200, 500),  # Whale
            "elephant": (4000, 8000),  # bat
        }

        self.instrument_freq_ranges = {
            "drum": (0, 630),
            "violin": (950, 6000),
            "trumpet": (300, 950),
            "piano": (100, 600),
        }
        self.uniform_freq_ranges = {}

        self.arrhythmia_ranges = {
            "af": (17, 150),
            "rbb": (0, 15),
            "syn": (0, 150),
        }

        # Adding Window modes to the combobox
        modes = ["Rectangular", "Hamming", "Hanning", "Gaussian"]
        combo_boxes = [
            self.ui.window_combobox,
            self.ui.window_combobox_2,
            self.ui.window_combobox_3,
            self.ui.window_combobox_4,
        ]

        for combo_box in combo_boxes:
            combo_box.addItems(modes)

        ### Connections between UI objects and the corresponding functions ###
        ######################################################################

        # plot connections for window combo boxes
        for combo_box in combo_boxes:
            combo_box.currentTextChanged.connect(self.plot_output_signal)

        # Signal-slot connections for sliders
        self.sliders = [
            self.ui.drum_slider,
            self.ui.violin_slider,
            self.ui.piano_slider,
            self.ui.trumpet_slider,
            self.ui.cat_slider,
            self.ui.dog_slider,
            self.ui.elephant_slider,
            self.ui.whale_slider,
            self.ui.rbb_slider,
            self.ui.af_slider,
            self.ui.syn_slider,
            self.ui.slider_one,
            self.ui.slider_two,
            self.ui.slider_three,
            self.ui.slider_four,
            self.ui.slider_five,
            self.ui.slider_six,
            self.ui.slider_seven,
            self.ui.slider_eight,
            self.ui.slider_nine,
            self.ui.slider_ten,
        ]
        # Set initial values for specific sliders
        for i, slider in enumerate(self.sliders):
            slider.valueChanged.connect(self.get_slider_values)

        self.audio_sliders = [
            self.ui.audio_volume_slider,
            self.ui.audio_volume_slider_2,
            self.ui.audio_volume_slider_3,
            self.ui.audio_volume_slider_4,
        ]

        for audio_slider in self.audio_sliders:
            audio_slider.setValue(50)
            audio_slider.setRange(0, 100)

        # Other signal-slot connections
        self.ui.actionOpen.triggered.connect(self.read_wav_file)
        self.timer.timeout.connect(self.plot_wav_file)
        self.ui.tabWidget.currentChanged.connect(self.tab_changed)
        self.choose_mode()
        self.ui.af_slider.setValue(50)
        self.ui.af_slider.setRange(0, 100)
        self.ui.rbb_slider.setValue(50)
        self.ui.rbb_slider.setRange(0, 100)
        self.ui.syn_slider.setValue(50)
        self.ui.syn_slider.setRange(0, 100)

    def tab_changed(self):
        self.choose_mode()
        # Clearing all lists, dictionaries and graphs when moving between tabs

        self.reset_all_sliders()

        self.data = []
        self.fft_data = []
        self.new_fft_data = []
        self.time = []
        self.frequencies = []
        self.sliced_data = {}
        self.slider_values = {}
        self.modified_data = {}

        self.input_signal_graph.clear()
        self.output_graph.clear()
        self.fft_graph.clear()

        self.figure.clear()
        self.canvas.draw()
        self.output_figure.clear()
        self.output_canvas.draw()

    def choose_mode(self):
        tab_index = self.ui.tabWidget.currentIndex()

        attributes = [
            (
                self.ui.input_signal_graph,
                self.ui.fft_graph,
                self.ui.output_signal_graph,
                self.ui.audio_select_radiobtn,
                self.ui.figure,
                self.ui.canvas,
                self.instrument_freq_ranges,
                self.ui.window_combobox,
                self.ui.figure_1,
                self.ui.play_pause_btn,
                self.ui.signal_speed_slider,
                self.ui.audio_volume_slider,
                self.ui.audio_mute_btn,
                self.ui.canvas_1,
                self.ui.hide_input_spectro_checkbox,
                self.ui.hide_output_spectro_checkbox,
                self.ui.zoomin_btn,
                self.ui.zoomout_btn,
                self.ui.reset_btn,
                self.ui.std_doubleSpinBox,
                self.ui.std_label,
            ),
            (
                self.ui.input_signal_graph_2,
                self.ui.fft_graph_2,
                self.ui.output_signal_graph_2,
                self.ui.audio_select_radiobtn_2,
                self.ui.figure_2,
                self.ui.canvas_2,
                self.animal_freq_ranges,
                self.ui.window_combobox_2,
                self.ui.figure_3,
                self.ui.play_pause_btn_2,
                self.ui.signal_speed_slider_2,
                self.ui.audio_volume_slider_2,
                self.ui.audio_mute_btn_2,
                self.ui.canvas_3,
                self.ui.hide_input_spectro_checkbox_2,
                self.ui.hide_output_spectro_checkbox_2,
                self.ui.zoomin_btn_2,
                self.ui.zoomout_btn_2,
                self.ui.reset_btn_2,
                self.ui.std_doubleSpinBox_2,
                self.ui.std_label_2,
            ),
            (
                self.ui.input_signal_graph_3,
                self.ui.fft_graph_3,
                self.ui.output_signal_graph_3,
                self.ui.audio_select_radiobtn_3,
                self.ui.figure_4,
                self.ui.canvas_4,
                self.uniform_freq_ranges,
                self.ui.window_combobox_3,
                self.ui.figure_5,
                self.ui.play_pause_btn_3,
                self.ui.signal_speed_slider_3,
                self.ui.audio_volume_slider_3,
                self.ui.audio_mute_btn_3,
                self.ui.canvas_5,
                self.ui.hide_input_spectro_checkbox_3,
                self.ui.hide_output_spectro_checkbox_3,
                self.ui.zoomin_btn_3,
                self.ui.zoomout_btn_3,
                self.ui.reset_btn_3,
                self.ui.std_doubleSpinBox_3,
                self.ui.std_label_3,
            ),
            (
                self.ui.input_signal_graph_4,
                self.ui.fft_graph_4,
                self.ui.output_signal_graph_4,
                self.ui.audio_select_radiobtn_4,
                self.ui.figure_6,
                self.ui.canvas_6,
                self.arrhythmia_ranges,
                self.ui.window_combobox_4,
                self.ui.figure_7,
                self.ui.play_pause_btn_4,
                self.ui.signal_speed_slider_4,
                self.ui.audio_volume_slider_4,
                self.ui.audio_mute_btn_4,
                self.ui.canvas_7,
                self.ui.hide_input_spectro_checkbox_4,
                self.ui.hide_output_spectro_checkbox_4,
                self.ui.zoomin_btn_4,
                self.ui.zoomout_btn_4,
                self.ui.reset_btn_4,
                self.ui.std_doubleSpinBox_4,
                self.ui.std_label_4,
            ),
        ]

        (
            self.input_signal_graph,
            self.fft_graph,
            self.output_graph,
            self.radio_btn,
            self.figure,
            self.canvas,
            self.mode,
            self.selected_mode,
            self.output_figure,
            self.play_pause_btn,
            self.speed_value,
            self.audio_volume_slider,
            self.audio_mute_btn,
            self.output_canvas,
            self.hide_input_spectro_checkbox,
            self.hide_output_spectro_checkbox,
            self.zoomin_btn,
            self.zoomout_btn,
            self.reset_btn,
            self.std_doubleSpinBox,
            self.std_label,
        ) = attributes[tab_index]

        ######################################## Connections ##########################################################

        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.zoomin_btn.clicked.connect(self.zoom_in)
        self.zoomout_btn.clicked.connect(self.zoom_out)
        self.reset_btn.clicked.connect(self.reset_signal)
        self.hide_input_spectro_checkbox.stateChanged.connect(
            self.hide_input_spectrograms
        )
        self.hide_output_spectro_checkbox.stateChanged.connect(
            self.hide_output_spectrograms
        )
        self.audio_volume_slider.valueChanged.connect(self.audio_level)
        self.speed_value.valueChanged.connect(self.speed_bar)
        self.speed_value.setRange(1, 4)
        self.audio_mute_btn.clicked.connect(self.mute_btn)
        self.audio_mute_btn.setCheckable(True)
        self.std_doubleSpinBox.valueChanged.connect(self.plot_gaussian)
        self.std_doubleSpinBox.setValue(10)
        self.std_doubleSpinBox.setRange(1, 1000000)
        self.std_doubleSpinBox.hide()
        self.std_label.hide()

    def read_wav_file(self):
        self.input_signal_graph.clear()
        options = QFileDialog.Options()
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Open WAV File",
            "",
            "WAV Files (*.wav)",
            options=options,
        )
        self.audio_time_signal = Signal(file_path=path)
        if self.audio_time_signal.file_path:
            self.data, self.sample_rate = librosa.load(self.audio_time_signal.file_path)
            self.duration = librosa.get_duration(y=self.data, sr=self.sample_rate)
            self.time = np.linspace(0, self.duration, len(self.data))

            # Check if the audio_select_radiobtn is checked
            if self.radio_btn.isChecked():
                # If checked, pass or continue
                pass
            else:
                # Start audio playback
                # Set up QMediaContent for audio playback
                content = QMediaContent(QUrl.fromLocalFile(path))
                self.media_player.setMedia(content)
                self.toggle_play_pause()

            self.calculate_plot_fft()
            # Start timer for real-time plotting
            self.index = 0  # position
            self.timer.start(1000)  # Interval to plot (Time taken to plot data)

    def plot_wav_file(self):
        output_graph = self.output_graph
        link_to_graph = self.input_signal_graph
        output_graph.setXLink(link_to_graph)
        output_graph.setYLink(link_to_graph)
        desired_duration = self.duration  # Duration in seconds
        step_size = round(
            len(self.data) / (desired_duration)
        )  # points plotted per second

        if self.index < len(self.data):
            x_min_limit = 0  # Set the minimum X-axis limit
            x_max_limit = max(self.time[self.index :])  # Set the maximum X-axis limit

            if self.index + step_size >= len(self.data):
                # If the next step exceeds the data length, plot the remaining data
                self.input_signal_graph.plot(
                    self.time[self.index :],
                    self.data[self.index :],
                    clear=False,
                )
                self.output_graph.plot(
                    self.time[: self.index + step_size],
                    self.data[: self.index + step_size],
                    clear=True,
                )
                self.timer.stop()  # Stop the timer when all data is plotted

            else:
                # Plot using the index to incrementally add data points
                self.input_signal_graph.plot(
                    self.time[: self.index + step_size],
                    self.data[: self.index + step_size],
                    clear=True,
                )
                self.output_graph.plot(
                    self.time[: self.index + step_size],
                    self.data[: self.index + step_size],
                    clear=True,
                )
                self.input_signal_graph.setLimits(
                    xMin=x_min_limit,
                    xMax=x_max_limit + 1,
                    yMin=-2,
                    yMax=5,
                )
                self.output_graph.setLimits(
                    xMin=x_min_limit,
                    xMax=x_max_limit + 1,
                    yMin=-2,
                    yMax=5,
                )

                self.index += step_size  # Increment by the calculated step size
        else:
            self.timer.stop()  # Stop the timer when all data is plotted
            self.calculate_plot_fft()

    def calculate_plot_fft(self):
        if len(self.data) > 0 and self.sample_rate != 0:
            self.fft_data = rfft(self.data)
            self.frequencies = rfftfreq(len(self.data)) * self.sample_rate
            fft_abs = np.abs(self.fft_data)[: len(self.frequencies) // 2]

            # Plotting the FFT
            # We take the one sided-positive self.frequencies, the negative sided one is symmetric anyways
            self.fft_graph.plot(
                self.frequencies[: len(self.frequencies) // 2],
                fft_abs,
            )
            # Plot the spectrogram
        self.plot_input_spectrogram()
        self.slice_fft_data()

    def plot_input_spectrogram(self):
        # Clear the selected canvas
        self.figure.clear()

        if len(self.data) > 0 and self.sample_rate != 0:
            # Create a new subplot for the spectrogram
            ax = self.figure.add_subplot(111)

            # Generate the spectrogram
            spec, freqs, times, im = ax.specgram(
                self.data,
                Fs=self.sample_rate,
                cmap=plt.cm.gist_heat,
            )

            # Set labels and title
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency")
            ax.set_title("Spectrogram")
            self.figure.colorbar(im, ax=ax, label="Intensity (dB)")

            # Show the spectrogram on the selected canvas
            self.canvas.draw()
            self.plot_output_spectrogram(self.data)

    def slice_fft_data(self):
        if self.ui.tabWidget.currentIndex() == 2:
            # Add vertical lines to represent frequency slices

            slice_names = [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
            ]
            # Clear any existing sliced data
            self.sliced_data = {}

            num_slices = 10
            positive_frequencies = self.frequencies[: len(self.frequencies) // 2]

            step_size = (
                max(positive_frequencies) - min(positive_frequencies)
            ) / num_slices

            positive_fft_data = np.abs(self.fft_data)[: len(self.frequencies) // 2]

            for i in range(num_slices):
                start_freq = i * step_size
                end_freq = (i + 1) * step_size

                start_idx = np.argmax(positive_frequencies >= start_freq)
                end_idx = np.argmax(positive_frequencies >= end_freq)
                key = slice_names[i]
                self.uniform_freq_ranges[key] = (
                    (positive_frequencies[start_idx]),
                    (positive_frequencies[end_idx]),
                )
                print(self.uniform_freq_ranges)

                # Store the slice ranges in a dictionary for later use
                self.sliced_data[key] = positive_fft_data[start_idx:end_idx]
        else:
            for key, freq_range in self.mode.items():
                start_idx = np.argmax(self.frequencies >= freq_range[0])
                end_idx = np.argmax(self.frequencies >= freq_range[1])
                self.window_length = end_idx - start_idx
                self.sliced_data[key] = self.fft_data[start_idx:end_idx]

    def get_slider_values(self):
        self.ui.af_slider.setValue(self.ui.af_slider.value())
        for key in self.sliced_data.keys():
            if self.ui.tabWidget.currentIndex() != 2:
                # Assuming the slider names in the UI follow the pattern 'animalName_slider' or 'instrumentName_slider'
                slider_name = f"{key}_slider"
                slider = getattr(self.ui, slider_name, 0)

            elif self.ui.tabWidget.currentIndex() == 2:
                slider_name = f"slider_{key}"
                slider = getattr(self.ui, slider_name, 0)

            if slider is not None:
                # Fetch the slider value and store it with the corresponding animal/instrument key
                self.slider_values[key] = slider.value() / 100
        print(self.slider_values)

    def apply_window_and_slider(self, window_mode):
        window_funcs = {
            "Rectangular": np.ones,
            "Hamming": np.hamming,
            "Hanning": np.hanning,
            "Gaussian": lambda size: signal.windows.gaussian(
                size, std=self.std_doubleSpinBox.value()
            ),
        }

        if window_mode in window_funcs:
            for key, segment in self.sliced_data.items():
                window_func = window_funcs[window_mode]
                window_size = len(segment)  # Calculate window size dynamically
                window = window_func(window_size) * self.slider_values.get(
                    key, 1
                )  # Retrieve slider value or default to 1
                modified_segment = segment * window  # Apply window to the segment
                self.modified_data[key] = modified_segment
        self.replace_data_slices()
        self.plot_window_functions()

    def plot_window_functions(self):
        self.std_doubleSpinBox.hide()
        self.fft_graph.clear()
        selected_mode = self.selected_mode.currentText()

        fft_abs = np.abs(self.fft_data)[: len(self.frequencies) // 2]

        # Plotting the FFT
        # We take the one sided-positive self.frequencies, the negative sided one is symmetric anyways
        self.fft_graph.plot(
            self.frequencies[: len(self.frequencies) // 2],
            fft_abs,
        )
        for key, segment in self.sliced_data.items():
            start_idx, end_idx = 0, 0
            if key in self.mode:
                start_idx = np.argmax(self.frequencies >= self.mode[key][0])
                end_idx = np.argmax(self.frequencies >= self.mode[key][1])
                self.window_length = end_idx - start_idx

            # Extract the corresponding frequency range for the slice
            slice_freq_range = self.frequencies[start_idx:end_idx]

            # Choose the window function based on the selected mode
            window_func = np.ones(self.window_length)  # Default to Rectangular window
            if selected_mode == "Hamming":
                window_func = np.hamming(self.window_length)
                self.std_doubleSpinBox.hide()
            elif selected_mode == "Hanning":
                window_func = np.hanning(self.window_length)
                self.std_doubleSpinBox.hide()
            elif selected_mode == "Gaussian":
                self.std_doubleSpinBox.show()
                window_func = signal.windows.gaussian(
                    self.window_length, std=self.std_doubleSpinBox.value()
                )

            # Multiply the window function by the maximum amplitude within the sliced data
            max_amplitude = np.max(np.abs(segment))  # Get the maximum amplitude
            adjusted_window = window_func * max_amplitude

            # Plot the window function against the frequency range on the fft_graph
            self.fft_graph.addItem(
                pg.InfiniteLine(
                    pos=self.frequencies[start_idx], angle=90, movable=False, pen="g"
                )
            )
            self.fft_graph.addItem(
                pg.InfiniteLine(
                    pos=self.frequencies[end_idx], angle=90, movable=False, pen="g"
                )
            )
            self.fft_graph.plot(
                slice_freq_range, adjusted_window, pen={"color": "r", "width": 3}
            )

    def replace_data_slices(self):
        self.new_fft_data = self.fft_data.copy()
        for key, modified_segment in self.modified_data.items():
            if key in self.mode:
                start_idx = np.argmax(self.frequencies >= self.mode[key][0])
                end_idx = np.argmax(self.frequencies >= self.mode[key][1])
                self.new_fft_data[start_idx:end_idx] = modified_segment

    def plot_output_signal(self):
        output_graph = self.output_graph
        link_to_graph = self.input_signal_graph
        window_mode = self.selected_mode.currentText()

        output_graph.clear()

        # entered here to do windowing and in there I call the replace then I save it in the self.new_fft_data which is the to be plotted data
        self.apply_window_and_slider(window_mode)
        inverted_data = irfft(self.new_fft_data / 2)

        self.save_wav_file()

        # Initialize the index for real-time plotting
        index = 0

        while index < len(inverted_data):
            # Plot using the index to incrementally add data points
            output_graph.plot(
                self.time[:index],
                np.real(inverted_data[:index]),
                clear=True,
            )

            index += 100  # Increment by the desired step size
            output_graph.setXLink(link_to_graph)
            output_graph.setYLink(link_to_graph)

            # Optional: You can add a delay to control the real-time plotting speed
            # QtCore.QCoreApplication.processEvents()
        self.plot_output_spectrogram(np.real(inverted_data))

    def plot_output_spectrogram(self, output_data):
        self.output_figure.clear()

        # Plotting the spectrogram
        ax = self.output_figure.add_subplot(111)
        spec, freqs, times, im = ax.specgram(
            output_data,
            Fs=self.sample_rate,
            cmap=plt.cm.gist_heat,
        )

        # Set labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_title("Spectrogram")
        self.output_figure.colorbar(im, ax=ax, label="Intensity (dB)")
        self.output_figure.canvas.draw()

    def save_wav_file(self):
        timestamp = time.strftime("%Y%m%d%H%M%S")  # Generate a timestamp
        filename = f"output_{timestamp}.wav"  # Use the timestamp in the filename

        modified_signal = irfft(self.new_fft_data) * (
            32767 / max(irfft(self.new_fft_data))
        )
        modified_signal = modified_signal.astype(np.int16)

        write(filename, self.sample_rate, modified_signal)
        self.output_audio(filename)

        print(f"Output WAV file '{filename}' is saved")

    def output_audio(self, output_file_path):
        try:
            # Check if the audio_select_radiobtn_3 is checked
            if self.radio_btn.isChecked():
                QtCore.QCoreApplication.processEvents()
                # Play the saved .wav file using QMediaPlayer
                media_content = QMediaContent(QUrl.fromLocalFile(output_file_path))
                self.media_player.setMedia(media_content)
                # Start audio playback
                self.toggle_play_pause()
            else:
                pass
        except Exception as e:
            print(f"Error during audio playback: {e}")

    ################################################### Buttons & Sliders #################################################################################

    def toggle_play_pause(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.timer.stop()  # Pause the real-time plot
            self.play_pause_btn.setIcon(QIcon("Images/play_button.png"))
            pass

        else:
            self.media_player.play()
            self.timer.start(
                1000
            )  # Resume the real-time plot (adjust interval as needed)
            self.play_pause_btn.setIcon(QIcon("Images/pause_button.png"))
            pass

    def plot_gaussian(self):
        self.plot_window_functions()
        self.plot_output_signal()

    def reset_signal(self):
        selected_mode = self.selected_mode.currentText()
        # Stop the timer and clear the graphs
        self.timer.stop()
        self.input_signal_graph.clear()
        self.output_graph.clear()

        # Reset the index for real-time plotting
        self.index = 0

        # Set media position to the start and start playback
        self.media_player.setPosition(0)
        self.media_player.play()

        # Restart the timer and plot the initial state
        self.timer.start(1000)  # Adjust the interval as needed
        self.plot_wav_file()
        # if selected_mode == "Gaussian":
        #

    def zoom_in(self):
        input_graph = self.input_signal_graph

        zoom_factor = 0.7
        # Adjust the visible range for zooming in

        x_range, y_range = input_graph.viewRange()
        center_x, center_y = sum(x_range) / 2, sum(y_range) / 2
        new_x_range = [
            (x_range[0] - center_x) * zoom_factor + center_x,
            (x_range[1] - center_x) * zoom_factor + center_x,
        ]
        new_y_range = [
            (y_range[0] - center_y) * zoom_factor + center_y,
            (y_range[1] - center_y) * zoom_factor + center_y,
        ]

        input_graph.setRange(xRange=new_x_range, yRange=new_y_range)

        pass

    def zoom_out(self):
        input_graph = self.input_signal_graph

        zoom_factor = 1.1
        x_range, y_range = input_graph.viewRange()
        center_x, center_y = sum(x_range) / 2, sum(y_range) / 2
        # Adjust the visible range for zooming in
        new_x_range = [
            (x_range[0] - center_x) * zoom_factor + center_x,
            (x_range[1] - center_x) * zoom_factor + center_x,
        ]
        new_y_range = [
            (y_range[0] - center_y) * zoom_factor + center_y,
            (y_range[1] - center_y) * zoom_factor + center_y,
        ]
        input_graph.setRange(xRange=new_x_range, yRange=new_y_range)
        pass

    def speed_bar(self):
        self.media_player.setPlaybackRate(self.speed_value.value())
        self.timer.start(
            self.speed_value.value()
        )  # Resume the real-time plot (adjust interval as needed)
        pass

    def audio_level(self):
        self.media_player.setVolume(
            self.audio_volume_slider.value()
        )  # Adjust volume as needed
        pass

    def mute_btn(self):
        muted = not self.audio_mute_btn.isChecked()
        self.original_volume = self.media_player.volume()
        icon = QtGui.QIcon()

        # Set the icon
        if muted:
            icon.addPixmap(
                QtGui.QPixmap("Images/Mute1.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off
            )
            self.media_player.setMuted(True)
        else:
            icon.addPixmap(
                QtGui.QPixmap("Images/unmute.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
            )
            self.media_player.setMuted(False)

        self.audio_mute_btn.setIcon(icon)

    def hide_input_spectrograms(self, state):
        if self.hide_input_spectro_checkbox.isChecked():
            self.canvas.setVisible(False)
        else:
            self.canvas.setVisible(True)

        pass

    def hide_output_spectrograms(self, state):
        if self.hide_output_spectro_checkbox.isChecked():
            self.output_canvas.setVisible(False)
        else:
            self.output_canvas.setVisible(True)

        pass

    def reset_all_sliders(self):
        for slider in self.sliders:
            if slider == self.ui.af_slider or self.ui.rbb_slider or self.ui.syn_slider:
                pass
            else:
                slider.setValue(0)
