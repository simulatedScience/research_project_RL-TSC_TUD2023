"""
This module provides functionality to animate traffic simulations using SUMO-GUI
with selected agents controlling traffic lights.

Authors: Sebastian Jost & GPT-4
"""
import os
import sys
import subprocess
import tempfile
import shutil
import yaml
import logging
from typing import Optional, Dict, Any
import tkinter as tk
from tkinter import messagebox, ttk, filedialog

# Add the project root to the path to import LibSignal modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from common.registry import Registry
from common import interface
from world.world_sumo import World
from agent.maxpressure import MaxPressureAgent
from agent.presslight import PressLightAgent
from agent.fixedtime import FixedTimeAgent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SUMO-GUI")


class SUMOGUIAnimator:
    """Class to handle SUMO-GUI animation with agent control."""
    
    def __init__(self):
        self.sumo_binary = self.find_sumo_binary()
        self.temp_dir = None
        
    def find_sumo_binary(self) -> Optional[str]:
        """Find SUMO binary on Windows system."""
        # Common SUMO installation paths on Windows
        possible_paths = [
            r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe",
            r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe",
            r"C:\sumo\bin\sumo-gui.exe",
            r"C:\Eclipse\Sumo\bin\sumo-gui.exe"
        ]
        
        # Check if sumo-gui is in PATH
        try:
            subprocess.run(["sumo-gui", "--version"], 
                         capture_output=True, check=True)
            return "sumo-gui"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Check common installation paths
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Check SUMO_HOME environment variable
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            sumo_gui_path = os.path.join(sumo_home, "bin", "sumo-gui.exe")
            if os.path.exists(sumo_gui_path):
                return sumo_gui_path
        
        logger.error("SUMO-GUI binary not found. Please install SUMO and ensure it's in your PATH.")
        return None
    
    def prepare_config_for_gui(self, config_path: str, agent_type: str, 
                              noise_settings: Optional[Dict] = None) -> str:
        """
        Prepare a SUMO configuration for GUI animation.
        
        Args:
            config_path (str): Path to original config file
            agent_type (str): Type of agent ('maxpressure', 'presslight', etc.)
            noise_settings (Dict, optional): Noise configuration
            
        Returns:
            str: Path to the prepared GUI config file
        """
        # Create temporary directory for GUI files
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="sumo_gui_")
        
        # Copy original config and modify for GUI
        gui_config_path = os.path.join(self.temp_dir, "gui_config.sumocfg")
        
        # Read original config
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Modify config for GUI
        config["gui"] = True
        config["name"] = f"gui_{agent_type}"
        
        # Set up output files in temp directory
        config["roadnetLogFile"] = os.path.join(self.temp_dir, "roadnet.json")
        config["replayLogFile"] = os.path.join(self.temp_dir, "replay.txt")
        
        # Write GUI config
        with open(gui_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return gui_config_path
    
    def register_command_settings(self, config_path: str, agent_type: str, noise_settings: Optional[Dict] = None):
        """Register command settings in Registry for agent initialization."""
        # Load default settings from base and agent configs
        base_config_path = os.path.join(project_root, "configs", "tsc", "base.yml")
        agent_config_path = os.path.join(project_root, "configs", "tsc", f"{agent_type}.yml")
        
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        try:
            with open(agent_config_path, 'r') as f:
                agent_config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Agent config file not found: {agent_config_path}")
            agent_config = {}
        
        # Create command settings for Registry
        cmd_settings = {
            "command": {
                "task": "tsc",
                "agent": agent_type,
                "world": "sumo",
                "network": os.path.splitext(os.path.basename(config_path))[0],
                "prefix": "gui_animation",
                "seed": 42,
                "failure_chance": noise_settings.get('failure_chance', 0.0) if noise_settings else 0.0,
                "tpr": noise_settings.get('true_positive_rate', 1.0) if noise_settings else 1.0,
                "fpr": noise_settings.get('false_positive_rate', 0.0) if noise_settings else 0.0,
                "interface": "libsumo",
                "delay_type": "apx",
            },
            "logger": {
                "path": self.temp_dir,
                "level": "INFO"
            },
            "world": {
                "world": "sumo",
                "config_file": config_path,
                "gui": True,
                "cityflow": {},
                "sumocfg": config_path,
                "roadnetFile": "",
                "flowFile": "",
            },
            "model": {
                "name": agent_type,
                "train_model": False,
                "test_model": True,
                "load_model": True,
            }
        }
        
        # Register settings in Registry
        interface.Command_Setting_Interface(cmd_settings)
        interface.World_param_Interface(cmd_settings)
        interface.ModelAgent_param_Interface(cmd_settings)
        
        return cmd_settings
    
    def create_agent(self, agent_type: str, world: World, 
                    noise_settings: Optional[Dict] = None):
        """Create and configure an agent for the simulation."""
        if agent_type == "maxpressure":
            agent = MaxPressureAgent(world, 0)
        elif agent_type == "presslight":
            agent = PressLightAgent(world, 0)
            if noise_settings:
                # Apply noise settings to the agent
                agent.FAILURE_CHANCE = noise_settings.get('failure_chance', 0.0)
                agent.TPR = noise_settings.get('true_positive_rate', 1.0)
                agent.FPR = noise_settings.get('false_positive_rate', 0.0)
                # Reload agent's noise configuration to ensure it's properly applied
                agent.reload_noise_config()
        elif agent_type == "fixedtime":
            agent = FixedTimeAgent(world, 0)
        else:
            logger.error(f"Unknown agent type: {agent_type}")
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return agent
    
    def run_gui_simulation(self, config_path: str, agent_type: str,
                          noise_settings: Optional[Dict] = None,
                          simulation_steps: int = 3600):
        """
        Run a SUMO-GUI simulation with agent control.
        
        Args:
            config_path (str): Path to SUMO config file
            agent_type (str): Type of agent to use
            noise_settings (Dict, optional): Noise configuration
            simulation_steps (int): Number of simulation steps
        """
        if not self.sumo_binary:
            logger.error("SUMO-GUI binary not found. Please install SUMO and ensure it's in your PATH.")
            return
        
        try:
            # Prepare GUI config
            gui_config = self.prepare_config_for_gui(config_path, agent_type, noise_settings)
            
            # Register settings in Registry for agent initialization
            cmd_settings = self.register_command_settings(gui_config, agent_type, noise_settings)
            
            # Initialize Registry for correct agent loading
            Registry.reset()
            Registry.register("world_mapping", {"setting": Registry})
            
            # Initialize world and agent
            world = World(gui_config)
            agent = self.create_agent(agent_type, world, noise_settings)
            
            # Reset world and agent
            world.reset()
            agent.reset()
            
            logger.info(f"Starting SUMO-GUI with {agent_type} agent. The simulation will run for {simulation_steps} steps.")
            print(f"Starting SUMO-GUI with {agent_type} agent. The simulation will run for {simulation_steps} steps.")
            
            # Run simulation with GUI
            for step in range(simulation_steps):
                # Get agent actions
                observations = agent.get_ob()
                current_phases = agent.get_phase()
                actions = agent.get_action(observations, current_phases, test=True)
                
                # Step the world
                world.step(actions)
                
                # Check if GUI window is still open
                if not world.eng.simulation.getMinExpectedNumber() > 0:
                    break
            
            logger.info("Simulation has finished or was closed.")
            print("Simulation has finished or was closed.")
            
        except Exception as e:
            logger.error(f"Failed to run simulation: {str(e)}")
            print(f"Error: Failed to run simulation: {str(e)}")
        finally:
            # Clean up
            self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {str(e)}")


class GUIAnimationDialog:
    """Dialog for configuring and launching GUI animation."""
    
    def __init__(self, parent=None):
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("SUMO-GUI Animation")
        self.root.configure(bg="#2e2e2e")
        self.root.geometry("500x450")
        
        self.animator = SUMOGUIAnimator()
        self.config_path = None
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create the dialog widgets."""
        # Title
        title_label = tk.Label(self.root, text="SUMO-GUI Animation Setup",
                              font=("Arial", 16, "bold"),
                              bg="#2e2e2e", fg="#ffffff")
        title_label.pack(pady=10)
        
        # Config file selection
        config_frame = tk.Frame(self.root, bg="#2e2e2e")
        config_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(config_frame, text="Config File:", bg="#2e2e2e", fg="#ffffff").pack(anchor=tk.W)
        
        config_path_frame = tk.Frame(config_frame, bg="#2e2e2e")
        config_path_frame.pack(fill=tk.X, pady=5)
        
        self.config_path_var = tk.StringVar()
        config_entry = tk.Entry(config_path_frame, textvariable=self.config_path_var,
                               bg="#3e3e3e", fg="#ffffff", relief=tk.FLAT)
        config_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_button = tk.Button(config_path_frame, text="Browse",
                                 command=self.browse_config,
                                 bg="#5e5e5e", fg="#ffffff", relief=tk.FLAT)
        browse_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Agent selection
        agent_frame = tk.Frame(self.root, bg="#2e2e2e")
        agent_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(agent_frame, text="Agent Type:", bg="#2e2e2e", fg="#ffffff").pack(anchor=tk.W)
        
        self.agent_var = tk.StringVar(value="maxpressure")
        agent_combo = ttk.Combobox(agent_frame, textvariable=self.agent_var,
                                  values=["maxpressure", "presslight", "fixedtime"],
                                  state="readonly")
        agent_combo.pack(fill=tk.X, pady=5)
        
        # Model selection
        model_frame = tk.Frame(self.root, bg="#2e2e2e")
        model_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(model_frame, text="Model Directory (optional):", bg="#2e2e2e", fg="#ffffff").pack(anchor=tk.W)
        
        model_path_frame = tk.Frame(model_frame, bg="#2e2e2e")
        model_path_frame.pack(fill=tk.X, pady=5)
        
        self.model_path_var = tk.StringVar()
        model_entry = tk.Entry(model_path_frame, textvariable=self.model_path_var,
                              bg="#3e3e3e", fg="#ffffff", relief=tk.FLAT)
        model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_model_button = tk.Button(model_path_frame, text="Browse",
                                      command=self.browse_model,
                                      bg="#5e5e5e", fg="#ffffff", relief=tk.FLAT)
        browse_model_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Noise settings (for presslight agent)
        noise_frame = tk.LabelFrame(self.root, text="Noise Settings (for PressLight)",
                                   bg="#2e2e2e", fg="#ffffff")
        noise_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Failure chance
        tk.Label(noise_frame, text="Failure Chance:", bg="#2e2e2e", fg="#ffffff").pack(anchor=tk.W)
        self.failure_chance_var = tk.DoubleVar(value=0.0)
        failure_scale = tk.Scale(noise_frame, from_=0.0, to=0.5, resolution=0.01,
                                orient=tk.HORIZONTAL, variable=self.failure_chance_var,
                                bg="#3e3e3e", fg="#ffffff", highlightthickness=0)
        failure_scale.pack(fill=tk.X, pady=2)
        
        # True positive rate
        tk.Label(noise_frame, text="True Positive Rate:", bg="#2e2e2e", fg="#ffffff").pack(anchor=tk.W)
        self.tpr_var = tk.DoubleVar(value=1.0)
        tpr_scale = tk.Scale(noise_frame, from_=0.5, to=1.0, resolution=0.01,
                            orient=tk.HORIZONTAL, variable=self.tpr_var,
                            bg="#3e3e3e", fg="#ffffff", highlightthickness=0)
        tpr_scale.pack(fill=tk.X, pady=2)
        
        # False positive rate
        tk.Label(noise_frame, text="False Positive Rate:", bg="#2e2e2e", fg="#ffffff").pack(anchor=tk.W)
        self.fpr_var = tk.DoubleVar(value=0.0)
        fpr_scale = tk.Scale(noise_frame, from_=0.0, to=0.3, resolution=0.01,
                            orient=tk.HORIZONTAL, variable=self.fpr_var,
                            bg="#3e3e3e", fg="#ffffff", highlightthickness=0)
        fpr_scale.pack(fill=tk.X, pady=2)
        
        # Simulation steps
        steps_frame = tk.Frame(self.root, bg="#2e2e2e")
        steps_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(steps_frame, text="Simulation Steps:", bg="#2e2e2e", fg="#ffffff").pack(anchor=tk.W)
        self.steps_var = tk.IntVar(value=3600)
        steps_entry = tk.Entry(steps_frame, textvariable=self.steps_var,
                              bg="#3e3e3e", fg="#ffffff", relief=tk.FLAT)
        steps_entry.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(self.root, textvariable=self.status_var, 
                               bg="#2e2e2e", fg="#aaaaaa", font=("Arial", 9))
        status_label.pack(fill=tk.X, padx=20, pady=2)
        
        # Buttons
        button_frame = tk.Frame(self.root, bg="#2e2e2e")
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        start_button = tk.Button(button_frame, text="Start Animation",
                                command=self.start_animation,
                                bg="#5e5e5e", fg="#ffffff", relief=tk.FLAT,
                                font=("Arial", 12, "bold"))
        start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        close_button = tk.Button(button_frame, text="Close",
                                command=self.root.destroy,
                                bg="#7e3e3e", fg="#ffffff", relief=tk.FLAT)
        close_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Default config paths
        self.set_default_config()
    
    def set_default_config(self):
        """Set a default config path."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_config = os.path.join(project_root, "configs", "sim", "sumo1x3.cfg")
        if os.path.exists(default_config):
            self.config_path_var.set(default_config)
    
    def browse_config(self):
        """Browse for a config file."""
        config_path = filedialog.askopenfilename(
            title="Select SUMO Config File",
            filetypes=[("Config files", "*.cfg"), ("All files", "*.*")]
        )
        if config_path:
            self.config_path_var.set(config_path)
    
    def browse_model(self):
        """Browse for a model directory."""
        model_path = filedialog.askdirectory(
            title="Select Model Directory"
        )
        if model_path:
            self.model_path_var.set(model_path)
    
    def start_animation(self):
        """Start the GUI animation."""
        config_path = self.config_path_var.get()
        if not config_path or not os.path.exists(config_path):
            self.status_var.set("Error: Please select a valid config file.")
            return
        
        agent_type = self.agent_var.get()
        
        # Prepare noise settings
        noise_settings = None
        if agent_type == "presslight":
            noise_settings = {
                'failure_chance': self.failure_chance_var.get(),
                'true_positive_rate': self.tpr_var.get(),
                'false_positive_rate': self.fpr_var.get()
            }
        
        simulation_steps = self.steps_var.get()
        
        # Update status
        self.status_var.set(f"Starting animation with {agent_type} agent...")
        self.root.update()
        
        # Start animation in a separate thread to prevent GUI freezing
        import threading
        animation_thread = threading.Thread(
            target=self.animator.run_gui_simulation,
            args=(config_path, agent_type, noise_settings, simulation_steps)
        )
        animation_thread.daemon = True
        animation_thread.start()
    
    def run(self):
        """Run the dialog."""
        self.root.mainloop()


def launch_gui_animation():
    """Launch the GUI animation dialog."""
    dialog = GUIAnimationDialog()
    dialog.run()


if __name__ == "__main__":
    launch_gui_animation()
