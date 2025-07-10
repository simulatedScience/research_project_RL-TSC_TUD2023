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
from typing import Optional, Dict, Any
import tkinter as tk
from tkinter import messagebox, ttk

# Add the project root to the path to import LibSignal modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from common.registry import Registry
from world.world_sumo import World
from agent.maxpressure import MaxPressureAgent
from agent.presslight import PressLightAgent


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
    
    def create_agent(self, agent_type: str, world: World, 
                    noise_settings: Optional[Dict] = None):
        """Create and configure an agent for the simulation."""
        if agent_type == "maxpressure":
            agent = MaxPressureAgent(world, 0)
        elif agent_type == "presslight":
            agent = PressLightAgent(world, 0)
            if noise_settings:
                # Apply noise settings to the agent
                agent.failure_chance = noise_settings.get('failure_chance', 0.0)
                agent.TPR = noise_settings.get('true_positive_rate', 1.0)
                agent.FPR = noise_settings.get('false_positive_rate', 0.0)
        else:
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
            messagebox.showerror("Error", 
                               "SUMO-GUI binary not found. Please install SUMO and ensure it's in your PATH.")
            return
        
        try:
            # Prepare GUI config
            gui_config = self.prepare_config_for_gui(config_path, agent_type, noise_settings)
            
            # Initialize world and agent
            Registry.register("world_mapping", {"setting": Registry})
            world = World(gui_config)
            agent = self.create_agent(agent_type, world, noise_settings)
            
            # Reset world and agent
            world.reset()
            agent.reset()
            
            messagebox.showinfo("Starting Simulation", 
                              f"Starting SUMO-GUI with {agent_type} agent.\n"
                              f"The simulation will run for {simulation_steps} steps.\n"
                              f"Close this dialog to begin.")
            
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
            
            messagebox.showinfo("Simulation Complete", 
                              "Simulation has finished or was closed.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run simulation: {str(e)}")
        finally:
            # Clean up
            self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            except Exception:
                pass


class GUIAnimationDialog:
    """Dialog for configuring and launching GUI animation."""
    
    def __init__(self, parent=None):
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("SUMO-GUI Animation")
        self.root.configure(bg="#2e2e2e")
        self.root.geometry("500x400")
        
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
                                  values=["maxpressure", "presslight"],
                                  state="readonly")
        agent_combo.pack(fill=tk.X, pady=5)
        
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
        
        # Buttons
        button_frame = tk.Frame(self.root, bg="#2e2e2e")
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        
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
        from tkinter import filedialog
        config_path = filedialog.askopenfilename(
            title="Select SUMO Config File",
            filetypes=[("Config files", "*.cfg"), ("All files", "*.*")]
        )
        if config_path:
            self.config_path_var.set(config_path)
    
    def start_animation(self):
        """Start the GUI animation."""
        config_path = self.config_path_var.get()
        if not config_path or not os.path.exists(config_path):
            messagebox.showerror("Error", "Please select a valid config file.")
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
