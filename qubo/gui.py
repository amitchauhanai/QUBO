import sys
import io
import os
import tempfile
import traceback
from typing import Optional

# Let `python gui.py` work when launched from inside the package directory.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_package_root = os.path.dirname(_this_dir)
if _package_root not in sys.path:
    sys.path.insert(0, _package_root)

# Matplotlib may try to create its cache under an unwritable home directory in
# sandboxed or packaged launches. Give it a stable writable location first.
_mpl_config_dir = os.path.join(tempfile.gettempdir(), 'qubo-matplotlib')
try:
    os.makedirs(_mpl_config_dir, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', _mpl_config_dir)
except Exception:
    pass

# Support either PyQt5 or PyQt6 depending on which is installed in the environment.
try:
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QFileDialog,
        QLabel,
        QListWidget,
        QPlainTextEdit,
        QSplitter,
        QDockWidget,
        QTabWidget,
        QSpinBox,
        QDoubleSpinBox,
        QComboBox,
        QCheckBox,
        QLineEdit,
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QFont
    QT_BINDING = 'PyQt5'
except Exception:
    from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QFileDialog,
        QLabel,
        QListWidget,
        QPlainTextEdit,
        QSplitter,
        QDockWidget,
        QTabWidget,
        QSpinBox,
        QDoubleSpinBox,
        QComboBox,
        QCheckBox,
        QLineEdit,
    )
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QFont
    QT_BINDING = 'PyQt6'
 
# Qt enum resolver usable for both PyQt5 and PyQt6
def _resolve_qt_enum(name: str):
    """Resolve a Qt enum name across PyQt5/PyQt6 compatibility layers.

    Returns the enum value or None if not found.
    """
    if 'Qt' in globals() and hasattr(Qt, name):
        return getattr(Qt, name)
    for attr_name in dir(Qt):
        try:
            attr = getattr(Qt, attr_name)
            if hasattr(attr, name):
                return getattr(attr, name)
        except Exception:
            continue
    return None

import matplotlib
if QT_BINDING == 'PyQt6':
    matplotlib.use('QtAgg')
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except Exception:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
else:
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
from qubo.simulator import StatevectorSimulator
from qubo import noise as qubo_noise
import json
from pathlib import Path
import threading
from qubo import copilot
from qubo import config as qubo_config

# Dark purple theme
STYLE = """
QWidget { background-color: #120022; color: #e6d6ff; }
QPlainTextEdit { background-color: #0f001b; color: #e6d6ff; border: 1px solid #2b003b; }
QLabel { color: #e6d6ff; }
QListWidget { background-color: #160028; color: #e6d6ff; border: 1px solid #2b003b; }
QPushButton { background-color: #220033; color: #e6d6ff; border: 1px solid #3a0050; padding: 6px; }
QPushButton:hover { background-color: #3a0050; }
QTabWidget::pane { background: #14001f; }
QTabBar::tab { background: #1a0028; color: #d9c8ff; padding: 6px; }
QTabBar::tab:selected { background: #2b0040; }
QSplitter { background: #120022; }
"""


class CodeEditor(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont('Menlo', 11))
        self.setPlainText(
            """from qubo.circuit import QuantumCircuit
from qubo.gates import H, X, Measure

qc = QuantumCircuit(2)
qc.add_gate(H(0))
qc.add_gate(X(1))
qc.add_gate(Measure(0))
"""
        )


class OutputPanel(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont('Menlo', 10))

    def write(self, text: str):
        self.appendPlainText(text)


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_subplot(111)

    def plot_state(self, state):
        self.ax.clear()
        arr = np.asarray(state)
        if np.iscomplexobj(arr):
            probs = (np.abs(arr) ** 2).real
        else:
            probs = arr.astype(float).real
        n = int((len(probs)).bit_length() - 1)
        labels = [format(i, f'0{n}b') for i in range(len(probs))]
        self.ax.bar(range(len(probs)), probs, color='#b892ff')
        self.ax.set_xticks(range(len(probs)))
        self.ax.set_xticklabels(labels, rotation=90, fontsize=8)
        self.ax.set_ylabel('Probability')
        self.figure.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    # Signals for thread-safe UI updates
    log_signal = pyqtSignal(str)
    chat_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('qubo — Quantum IDE')
        self.resize(1100, 700)

        # Qt enum compatibility (PyQt5 vs PyQt6 moved enums)
        def _qt_enum(name):
            if hasattr(Qt, name):
                return getattr(Qt, name)
            # try nested enum containers on Qt
            for attr_name in dir(Qt):
                try:
                    attr = getattr(Qt, attr_name)
                    if hasattr(attr, name):
                        return getattr(attr, name)
                except Exception:
                    continue
            return None

        # Use default splitter orientation (typically Horizontal) to avoid
        # depending on Qt enum naming across PyQt5/PyQt6 differences.
        central = QSplitter()

        # Left: gates + preview
        left = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.addWidget(QLabel('Gates'))
        self.gate_list = QListWidget()
        for g in ['H', 'X', 'CNOT', 'RZ', 'Measure']:
            self.gate_list.addItem(g)
        left_layout.addWidget(self.gate_list)
        left_layout.addWidget(QLabel('Circuit preview'))
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        left_layout.addWidget(self.preview)

        left.setLayout(left_layout)

        central.addWidget(left)

        # Right: editor + tabs
        right = QWidget()
        right_layout = QVBoxLayout()
        toolbar = QHBoxLayout()
        self.open_btn = QPushButton('Open')
        self.save_btn = QPushButton('Save')
        self.run_btn = QPushButton('Run')
        toolbar.addWidget(self.open_btn)
        toolbar.addWidget(self.save_btn)
        toolbar.addStretch()
        toolbar.addWidget(QLabel('Simulator:'))
        self.simulator_combo = QComboBox()
        self.simulator_combo.addItems(['Statevector', 'Density Matrix'])
        toolbar.addWidget(self.simulator_combo)
        toolbar.addWidget(QLabel('Shots:'))
        self.shots_spin = QSpinBox()
        self.shots_spin.setRange(1, 100000)
        self.shots_spin.setValue(1024)
        toolbar.addWidget(self.shots_spin)
        toolbar.addWidget(QLabel('Noise:'))
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(['None', 'Bit-flip', 'Phase-flip', 'Depolarizing', 'Amplitude Damping'])
        toolbar.addWidget(self.noise_combo)
        toolbar.addWidget(QLabel('Strength:'))
        self.noise_prob_spin = QDoubleSpinBox()
        self.noise_prob_spin.setRange(0.0, 1.0)
        self.noise_prob_spin.setSingleStep(0.01)
        self.noise_prob_spin.setValue(0.01)
        toolbar.addWidget(self.noise_prob_spin)
        toolbar.addWidget(self.run_btn)
        right_layout.addLayout(toolbar)

        self.editor = CodeEditor()
        right_layout.addWidget(self.editor, stretch=3)

        self.tabs = QTabWidget()
        self.output = OutputPanel()
        self.sim_tab = QWidget()
        sim_layout = QVBoxLayout()
        self.sim_output = OutputPanel()
        sim_layout.addWidget(QLabel('Simulator Output'))
        sim_layout.addWidget(self.sim_output)
        self.sim_tab.setLayout(sim_layout)

        self.vis_tab = QWidget()
        vis_layout = QVBoxLayout()
        self.canvas = MatplotlibCanvas()
        vis_layout.addWidget(self.canvas)
        self.vis_tab.setLayout(vis_layout)

        self.tabs.addTab(self.output, 'Output')
        self.tabs.addTab(self.sim_tab, 'Simulator')
        # Copilot tab for ML assistance and short experiments
        self.copilot_tab = QWidget()
        copilot_layout = QVBoxLayout()
        self.copilot_out = QPlainTextEdit()
        self.copilot_out.setReadOnly(True)
        # connect log signal for thread-safe updates
        self.log_signal.connect(self.copilot_out.appendPlainText)
        # Gemini key entry
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel('Gemini Key:'))
        self.gemini_key_edit = QLineEdit()
        self.gemini_key_edit.setPlaceholderText('Paste Gemini API key (optional)')
        # hide key by default; handle PyQt5 vs PyQt6 enum differences
        try:
            # PyQt5
            self.gemini_key_edit.setEchoMode(QLineEdit.Password)
        except Exception:
            try:
                # PyQt6: EchoMode nested enum
                self.gemini_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            except Exception:
                pass
        key_layout.addWidget(self.gemini_key_edit)
        self.save_key_btn = QPushButton('Save Key')
        self.clear_key_btn = QPushButton('Clear Key')
        key_layout.addWidget(self.save_key_btn)
        key_layout.addWidget(self.clear_key_btn)
        copilot_layout.addLayout(key_layout)

        btn_layout = QHBoxLayout()
        self.detect_btn = QPushButton('Detect Backends')
        self.run_ml_btn = QPushButton('Run Copilot Example')
        btn_layout.addWidget(self.detect_btn)
        btn_layout.addWidget(self.run_ml_btn)
        copilot_layout.addLayout(btn_layout)
        # Chat moved to a flexible dock on the right side
        copilot_layout.addWidget(self.copilot_out)
        self.copilot_tab.setLayout(copilot_layout)
        self.tabs.addTab(self.copilot_tab, 'Copilot')
        self.tabs.addTab(self.vis_tab, 'Visualizer')

        right_layout.addWidget(self.tabs, stretch=2)
        right.setLayout(right_layout)

        central.addWidget(right)
        central.setStretchFactor(0, 1)
        central.setStretchFactor(1, 3)

        self.setCentralWidget(central)

        # create chat dock (right side)
        self.create_chat_dock()
        # create terminal dock for direct agent interaction
        self.create_terminal_dock()
        # create properties dock (VS Code-like properties) for quantum elements
        self.create_properties_dock()

        # connections
        self.open_btn.clicked.connect(self.open_file)
        self.save_btn.clicked.connect(self.save_file)
        self.run_btn.clicked.connect(self.run_all)
        self.gate_list.itemDoubleClicked.connect(self.insert_gate)
        self.gate_list.itemClicked.connect(self.handle_property_select)
        self.detect_btn.clicked.connect(self.handle_detect_backends)
        self.run_ml_btn.clicked.connect(self.handle_run_copilot_example)
        self.save_key_btn.clicked.connect(self.handle_save_gemini_key)
        self.clear_key_btn.clicked.connect(self.handle_clear_key)

        # config path
        self._config_path = Path.cwd() / '.qubo_config.json'
        self._enabled_extras = set()
        self._auto_select = False
        self.load_config()

        self.setStyleSheet(STYLE)

    def _add_dock(self, dock, preferred_names=None):
        """Add a dock widget trying several Qt enum names for compatibility.

        preferred_names: list of enum attribute names to try on Qt.
        """
        if preferred_names is None:
            preferred_names = ['RightDockWidgetArea', 'RightDockArea', 'LeftDockWidgetArea', 'LeftDockArea']
        # try preferred named enums first
        for name in preferred_names:
            try:
                area = _resolve_qt_enum(name)
                if area is not None:
                    try:
                        self.addDockWidget(area, dock)
                        return
                    except Exception:
                        # try next
                        continue
            except Exception:
                continue
        # fallback: try DockWidgetArea attribute
        try:
            if hasattr(Qt, 'DockWidgetArea'):
                try:
                    self.addDockWidget(Qt.DockWidgetArea, dock)
                    return
                except Exception:
                    pass
        except Exception:
            pass
        # last resort: try calling without area (some bindings accept it)
        try:
            self.addDockWidget(dock)
        except Exception:
            # give up silently
            pass

        # connections
        self.open_btn.clicked.connect(self.open_file)
        self.save_btn.clicked.connect(self.save_file)
        self.run_btn.clicked.connect(self.run_all)
        self.gate_list.itemDoubleClicked.connect(self.insert_gate)
        self.gate_list.itemClicked.connect(self.handle_property_select)
        self.detect_btn.clicked.connect(self.handle_detect_backends)
        self.run_ml_btn.clicked.connect(self.handle_run_copilot_example)
        self.save_key_btn.clicked.connect(self.handle_save_gemini_key)
        self.clear_key_btn.clicked.connect(self.handle_clear_key)

        # config path
        self._config_path = Path.cwd() / '.qubo_config.json'
        self._enabled_extras = set()
        self._auto_select = False
        self.load_config()

        self.setStyleSheet(STYLE)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Python File', '', 'Python Files (*.py)')
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.editor.setPlainText(f.read())
        except Exception as e:
            self.output.write('Error opening file: ' + str(e))

    def save_file(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save Python File', '', 'Python Files (*.py)')
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.editor.toPlainText())
            self.output.write(f'Saved: {path}')
        except Exception as e:
            self.output.write('Error saving file: ' + str(e))

    def insert_gate(self, item):
        g = item.text()
        cursor = self.editor.textCursor()
        if g == 'CNOT':
            cursor.insertText("qc.add_gate('CNOT', targets=[0,1])\n")
        elif g == 'RZ':
            cursor.insertText("qc.add_gate('RZ', targets=[0], params=[3.1415])\n")
        else:
            cursor.insertText(f"qc.add_gate('{g}', targets=[0])\n")

    def run_all(self):
        code = self.editor.toPlainText()
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        try:
            ns = {}
            exec(code, ns)
            if 'qc' in ns:
                noise_choice = self.noise_combo.currentText()
                sim_choice = self.simulator_combo.currentText()
                noise_hook = None
                noise_strength = float(self.noise_prob_spin.value())
                if noise_choice != 'None':
                    mapping = {
                        'Bit-flip': qubo_noise.bit_flip,
                        'Phase-flip': qubo_noise.phase_flip,
                        'Depolarizing': qubo_noise.depolarizing,
                        'Amplitude Damping': qubo_noise.amplitude_damping,
                    }
                    func = mapping.get(noise_choice)
                    if func:
                        def _hook(state, gate, _f=func, _p=noise_strength):
                            for t in gate.targets:
                                state = _f(state, _p, t)
                            return state
                        noise_hook = _hook

                shots = int(self.shots_spin.value())
                if sim_choice == 'Density Matrix':
                    from qubo.density import DensityMatrixSimulator
                    sim = DensityMatrixSimulator(ns['qc'])
                    res = sim.run(shots=shots, noise_hook=noise_hook)
                else:
                    sim = StatevectorSimulator(ns['qc'], noise_hook=noise_hook, backend='numpy')
                    res = sim.run(shots=shots)
                if isinstance(res, dict):
                    buf_out.write(str(res) + '\n')
                    self.sim_output.setPlainText(str(res))
                    total = sum(res.values())
                    approx = [res.get(format(i, f'0{ns["qc"].num_qubits}b'), 0) / total for i in range(2 ** ns['qc'].num_qubits)]
                    self.canvas.plot_state(np.array(approx))
                else:
                    if hasattr(res, 'ndim') and res.ndim == 2:
                        probs = np.real(np.diag(res))
                        labels = [format(i, f'0{ns["qc"].num_qubits}b') for i in range(len(probs))]
                        out = {labels[i]: float(probs[i]) for i in range(len(probs))}
                        self.sim_output.setPlainText(str(out))
                        self.canvas.plot_state(probs)
                        buf_out.write('Density matrix probabilities:\n')
                        for label, value in out.items():
                            buf_out.write(f'{label} {value}\n')
                    else:
                        self.canvas.plot_state(res)
                        buf_out.write('Statevector length: %d\n' % len(res))
                self.tabs.setCurrentIndex(1)
            else:
                buf_out.write("No 'qc' QuantumCircuit found in code.\n")
        except Exception:
            buf_err.write(traceback.format_exc())

        out_text = buf_out.getvalue()
        err_text = buf_err.getvalue()
        if out_text:
            self.output.write(out_text)
        if err_text:
            self.output.write(err_text)

    def handle_detect_backends(self):
        try:
            backends = copilot.detect_backends()
            self.log_signal.emit(str(backends))
        except Exception as e:
            self.log_signal.emit('Detect error: ' + str(e))

    def handle_run_copilot_example(self):
        def _run():
            self.log_signal.emit('Starting Copilot example...')
            try:
                hist = copilot.example_training_run()
                self.log_signal.emit('Result:')
                self.log_signal.emit(str(hist))
            except Exception as e:
                self.log_signal.emit('Error: ' + str(e))

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def load_config(self):
        try:
            if self._config_path.exists():
                data = json.loads(self._config_path.read_text(encoding='utf-8'))
                self._enabled_extras = set(data.get('enabled_extras', []))
                self._auto_select = data.get('auto_select', False)
                # load gemini key if present
                gemini = data.get('gemini_key')
                if gemini:
                    try:
                        self.gemini_key_edit.setText(gemini)
                    except Exception:
                        pass
        except Exception:
            pass

    def save_config(self):
        try:
            data = {'enabled_extras': list(self._enabled_extras), 'auto_select': self._auto_select}
            # include gemini key if set
            try:
                key = self.gemini_key_edit.text().strip()
                if key:
                    data['gemini_key'] = key
            except Exception:
                pass
            self._config_path.write_text(json.dumps(data), encoding='utf-8')
        except Exception:
            pass

    def handle_save_gemini_key(self):
        try:
            key = self.gemini_key_edit.text().strip()
            data = {}
            if self._config_path.exists():
                try:
                    data = json.loads(self._config_path.read_text(encoding='utf-8'))
                except Exception:
                    data = {}
            if key:
                data['gemini_key'] = key
            else:
                data.pop('gemini_key', None)
            self._config_path.write_text(json.dumps(data), encoding='utf-8')
            self.log_signal.emit('Gemini key saved to ' + str(self._config_path))
        except Exception as e:
            self.log_signal.emit('Error saving Gemini key: ' + str(e))

    def handle_clear_key(self):
        try:
            if self._config_path.exists():
                try:
                    data = json.loads(self._config_path.read_text(encoding='utf-8'))
                except Exception:
                    data = {}
                data.pop('gemini_key', None)
                self._config_path.write_text(json.dumps(data), encoding='utf-8')
            # clear UI field
            try:
                self.gemini_key_edit.setText('')
            except Exception:
                pass
            self.log_signal.emit('Gemini key cleared from config')
        except Exception as e:
            self.log_signal.emit('Error clearing Gemini key: ' + str(e))

    def handle_send_chat(self):
        msg = self.chat_input.text().strip()
        if not msg:
            return
        # show user message
        self.chat_signal.emit('You: ' + msg)
        self.chat_input.clear()

        def _call():
            self.chat_signal.emit('Assistant: (thinking...)')
            try:
                model = self.model_combo.currentText()
                res = copilot.call_gemini(msg, model=model)
                # pretty-print response
                if isinstance(res, dict):
                    text = res.get('candidates') or res.get('output') or res.get('text') or str(res)
                else:
                    text = str(res)
                self.chat_signal.emit('Assistant: ' + str(text))
            except Exception as e:
                self.chat_signal.emit('Assistant error: ' + str(e))

        t = threading.Thread(target=_call, daemon=True)
        t.start()

    def create_chat_dock(self):
        """Create a resizable chat dock on the right side with model selector and input."""
        dock = QDockWidget('Copilot Chat', self)
        right_area = _resolve_qt_enum('RightDockWidgetArea') or _resolve_qt_enum('RightDockArea') or _resolve_qt_enum('DockWidgetArea')
        left_area = _resolve_qt_enum('LeftDockWidgetArea') or _resolve_qt_enum('LeftDockArea') or _resolve_qt_enum('DockWidgetArea')
        if right_area is not None and left_area is not None:
            dock.setAllowedAreas(right_area | left_area)
        else:
            try:
                dock.setAllowedAreas(Qt.DockWidgetArea)
            except Exception:
                # fallback: no-op if we cannot set allowed areas
                pass
        w = QWidget()
        layout = QVBoxLayout()

        self.chat_history = QPlainTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setMinimumHeight(200)
        layout.addWidget(self.chat_history)
        # connect chat signal
        self.chat_signal.connect(self.chat_history.appendPlainText)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel('Model:'))
        self.model_combo = QComboBox()
        self.model_combo.addItems(['models/gemini-1.0', 'models/gemini-1.1'])
        ctrl.addWidget(self.model_combo)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText('Type a message and press Send')
        ctrl.addWidget(self.chat_input)
        self.chat_send_btn = QPushButton('Send')
        ctrl.addWidget(self.chat_send_btn)
        layout.addLayout(ctrl)

        w.setLayout(layout)
        dock.setWidget(w)
        # add dock using compatibility helper
        self._add_dock(dock, preferred_names=['RightDockWidgetArea', 'RightDockArea'])
        # connect send button
        self.chat_send_btn.clicked.connect(self.handle_send_chat)

    def create_terminal_dock(self):
        """Create a simple terminal-like dock that sends typed lines to the agent."""
        dock = QDockWidget('Qubo Terminal', self)
        right_area = _resolve_qt_enum('RightDockWidgetArea') or _resolve_qt_enum('RightDockArea') or _resolve_qt_enum('DockWidgetArea')
        if right_area is not None:
            try:
                dock.setAllowedAreas(right_area)
            except Exception:
                pass

        w = QWidget()
        layout = QVBoxLayout()

        self.terminal_output = QPlainTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setMinimumHeight(120)
        layout.addWidget(self.terminal_output)

        ctrl = QHBoxLayout()
        self.terminal_input = QLineEdit()
        self.terminal_input.setPlaceholderText('Enter command for Copilot agent')
        ctrl.addWidget(self.terminal_input)
        self.terminal_send_btn = QPushButton('Send')
        ctrl.addWidget(self.terminal_send_btn)
        layout.addLayout(ctrl)

        w.setLayout(layout)
        dock.setWidget(w)
        # add dock using compatibility helper
        self._add_dock(dock, preferred_names=['RightDockWidgetArea', 'RightDockArea'])

        # connect actions
        self.terminal_send_btn.clicked.connect(self.handle_send_terminal)
        try:
            self.terminal_input.returnPressed.connect(self.handle_send_terminal)
        except Exception:
            # PyQt6 may use different signal name; ignore if not present
            pass

    def handle_send_terminal(self):
        msg = self.terminal_input.text().strip()
        if not msg:
            return
        # echo to terminal output
        try:
            self.terminal_output.appendPlainText('> ' + msg)
        except Exception:
            pass
        self.terminal_input.clear()

        def _call():
            try:
                self.terminal_output.appendPlainText('(assistant thinking...)')
                model = getattr(self, 'model_combo', None)
                m = model.currentText() if model is not None else None
                res = copilot.call_gemini(msg, model=m) if m else copilot.call_gemini(msg)
                if isinstance(res, dict):
                    text = res.get('candidates') or res.get('output') or res.get('text') or str(res)
                else:
                    text = str(res)
                self.terminal_output.appendPlainText('Assistant: ' + str(text))
            except Exception as e:
                self.terminal_output.appendPlainText('Assistant error: ' + str(e))

        t = threading.Thread(target=_call, daemon=True)
        t.start()

    def create_properties_dock(self):
        """Create a VS Code-like Properties dock for quantum gates/circuits."""
        dock = QDockWidget('Properties', self)
        left_area = _resolve_qt_enum('LeftDockWidgetArea') or _resolve_qt_enum('LeftDockArea') or _resolve_qt_enum('DockWidgetArea')
        if left_area is not None:
            try:
                dock.setAllowedAreas(left_area)
            except Exception:
                pass

        w = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel('Selected'))
        self.prop_selected_label = QLabel('(none)')
        layout.addWidget(self.prop_selected_label)

        layout.addWidget(QLabel('Target Qubit'))
        self.prop_target_spin = QSpinBox()
        self.prop_target_spin.setRange(0, 32)
        self.prop_target_spin.setValue(0)
        layout.addWidget(self.prop_target_spin)

        layout.addWidget(QLabel('Params (comma-separated)'))
        self.prop_params_edit = QLineEdit()
        layout.addWidget(self.prop_params_edit)

        self.prop_apply_btn = QPushButton('Apply / Insert')
        layout.addWidget(self.prop_apply_btn)

        w.setLayout(layout)
        dock.setWidget(w)
        # add dock using compatibility helper
        self._add_dock(dock, preferred_names=['LeftDockWidgetArea', 'LeftDockArea'])

        self.prop_apply_btn.clicked.connect(self.handle_apply_properties)

    def handle_property_select(self, item):
        try:
            name = item.text()
            self.prop_selected_label.setText(name)
            # sensible defaults
            self.prop_target_spin.setValue(0)
            self.prop_params_edit.setText('')
        except Exception:
            pass

    def handle_apply_properties(self):
        try:
            gate = self.prop_selected_label.text()
            target = int(self.prop_target_spin.value())
            params = self.prop_params_edit.text().strip()
            cursor = self.editor.textCursor()
            if params:
                # clean up params
                p = ','.join([s.strip() for s in params.split(',') if s.strip()])
                cursor.insertText(f"qc.add_gate('{gate}', targets=[{target}], params=[{p}])\n")
            else:
                cursor.insertText(f"qc.add_gate('{gate}', targets=[{target}])\n")
            self.output.write(f'Inserted gate {gate} target={target} params={params}')
        except Exception as e:
            self.output.write('Error applying properties: ' + str(e))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    # Normalize QApplication run method across PyQt5/PyQt6
    exec_fn = getattr(app, 'exec', None) or getattr(app, 'exec_', None)
    if not exec_fn:
        sys.exit(1)
    return_code = exec_fn()
    sys.exit(return_code)


if __name__ == '__main__':
    main()
