from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import (
    Header, Footer, Input, Label, TextArea, Button, 
    RichLog, Select
)
from textual.binding import Binding
import json
import asyncio
import httpx

class OllamaUI(App):
    """A Textual app to interact with Ollama API."""
    
    CSS_PATH = "ollama_ui.tcss"
    
    # Key bindings
    BINDINGS = [
        Binding("d", "toggle_dark", "Toggle dark mode", priority=True),
        Binding("q", "quit", "Quit", priority=True),
        Binding("r", "refresh_models", "Refresh models", priority=True),
    ]
    
    def __init__(self):
        super().__init__()
        self.api_base = "http://localhost:11434"
        self.last_responses = []  # Store the last set of responses
        self.last_prompts = {"system": "", "user": ""}  # Store the last prompts
        
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        
        # Main content area
        with Vertical(id="main"):
            # Model selection
            with Container(id="model_selection"):
                yield Label("Model Name")
                yield Select(
                    [],  # Empty initially
                    prompt="Select a model",
                    id="model_input",
                )
                yield Button("↻", id="refresh", variant="primary")
            
            # System prompt
            yield Label("System Prompt")
            yield TextArea(id="system_prompt")
            
            # User prompt
            yield Label("User Prompt")
            yield TextArea(id="user_prompt")
            
            # Parameters
            with Container(id="parameters"):
                yield Label("Temperature")
                yield Select(
                    [(str(round(t/10, 1)), str(round(t/10, 1))) for t in range(0, 11)],
                    value="0.7",
                    id="temperature",
                    prompt="Select temperature"
                )
                
                yield Label("Max Tokens")
                yield Select(
                    [("500", "500"), ("1000", "1000"), ("1500", "1500"), 
                     ("2000", "2000"), ("2048", "2048"), ("3000", "3000"), 
                     ("4000", "4000")],
                    value="2048",
                    id="max_tokens",
                    prompt="Select max tokens"
                )
                
                yield Label("Context Window")
                yield Select(
                    [("1024", "1024"), ("2048", "2048"), ("4096", "4096"), 
                     ("8192", "8192")],
                    value="4096",
                    id="context_window",
                    prompt="Select context window"
                )
                
                yield Label("Concurrency Count")
                yield Select(
                    [("5", "5"), ("10", "10"), ("20", "20"), 
                     ("50", "50"), ("100", "100")],
                    value="5",
                    id="concurrency",
                    prompt="Select concurrency"
                )

                yield Label("Max Concurrent Requests")
                yield Select(
                    [("1", "1"), ("2", "2"), ("3", "3"), ("4", "4")],
                    value="2",
                    id="max_concurrent",
                    prompt="Select max concurrent requests"
                )
            
            # Buttons
            with Container(id="buttons"):
                yield Button("Run", id="run", variant="success")
                yield Button("Export CSV", id="export", variant="default")
            
            # Output area
            yield Label("Output")
            yield RichLog(id="output", wrap=True, markup=True, highlight=True)
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """Event handler called when app is mounted."""
        self.title = "Ollama API UI"
        
        # Set default text for TextAreas
        system_prompt = self.query_one("#system_prompt", TextArea)
        user_prompt = self.query_one("#user_prompt", TextArea)
        
        system_prompt.placeholder = "Enter system prompt..."
        user_prompt.placeholder = "Enter user prompt..."
        
        # Fetch models on startup
        await self.refresh_models()
    
    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark
    
    def action_refresh_models(self) -> None:
        """Action to refresh models list."""
        self.run_worker(self.refresh_models())
    
    async def refresh_models(self) -> None:
        """Refresh the list of available models."""
        try:
            output = self.query_one("#output", RichLog)
            model_select = self.query_one("#model_input", Select)
            
            output.write("[bold blue]Fetching available models...[/]")
            
            async with httpx.AsyncClient(base_url=self.api_base) as client:
                response = await client.get("/api/tags")
                if response.status_code == 200:
                    models_data = response.json()
                    models = [(model["name"], model["name"]) for model in models_data.get("models", [])]
                    
                    # Update the Select widget with new model options
                    model_select.set_options(models)
                    
                    # Display models in output
                    output.write("[bold green]Available Models:[/]")
                    for model in models_data.get("models", []):
                        size_mb = model["size"] / (1024 * 1024)  # Convert to MB
                        output.write(f"• {model['name']} ({size_mb:.1f} MB)")
                else:
                    output.write("[bold red]Error fetching models[/]")
        except Exception as e:
            output = self.query_one("#output", RichLog)
            output.write(f"[bold red]Error: {str(e)}[/]")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "refresh":
            await self.refresh_models()
        elif button_id == "run":
            await self.run_prompts()
        elif button_id == "export":
            await self.export_results()
    
    async def run_prompts(self) -> None:
        """Run prompts with specified concurrency."""
        try:
            # Get all input values
            model = self.query_one("#model_input", Select).value
            system_prompt = self.query_one("#system_prompt", TextArea).text
            user_prompt = self.query_one("#user_prompt", TextArea).text
            
            # Store prompts for export
            self.last_prompts = {
                "system": system_prompt,
                "user": user_prompt
            }

            try:
                temperature = float(self.query_one("#temperature", Select).value)
                max_tokens = int(self.query_one("#max_tokens", Select).value)
                concurrency = int(self.query_one("#concurrency", Select).value)
                max_concurrent = int(self.query_one("#max_concurrent", Select).value)
                
                # Validate ranges
                if not (0 <= temperature <= 2):
                    raise ValueError("Temperature must be between 0 and 2")
                if max_tokens < 1:
                    raise ValueError("Max tokens must be positive")
                if not (1 <= concurrency <= 100):
                    raise ValueError("Total requests must be between 1 and 100")
                if not (1 <= max_concurrent <= 4):
                    raise ValueError("Max concurrent requests must be between 1 and 4")
                
            except ValueError as e:
                output = self.query_one("#output", RichLog)
                output.write(f"[bold red]Invalid parameter: {str(e)}[/]")
                return
            
            output = self.query_one("#output", RichLog)
            
            if not model or not user_prompt:
                output.write("[bold red]Please select a model and enter a user prompt[/]")
                return
            
            output.write(f"[bold blue]Running {concurrency} total requests with {max_concurrent} concurrent requests at a time...[/]")
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def limited_generate():
                async with semaphore:
                    return await self.generate_completion(model, system_prompt, user_prompt, temperature, max_tokens)
            
            # Create tasks for execution
            tasks = [limited_generate() for _ in range(concurrency)]
            
            # Run tasks with concurrency limit
            responses = await asyncio.gather(*tasks)
            
            # Store responses for export
            self.last_responses = responses
            
            # Display results
            output.write("\n[bold green]Results:[/]")
            for i, response in enumerate(responses, 1):
                output.write(f"\n[bold]Response {i}:[/]")
                output.write(response)
            
            # Simple variance analysis
            unique_responses = len(set(responses))
            output.write(f"\n[bold]Variance Analysis:[/]")
            output.write(f"Unique responses: {unique_responses} out of {concurrency}")
            
        except Exception as e:
            output = self.query_one("#output", RichLog)
            output.write(f"[bold red]Error: {str(e)}[/]")
    
    async def generate_completion(self, model: str, system_prompt: str, user_prompt: str, 
                                temperature: float, max_tokens: int) -> str:
        """Generate a completion from Ollama API."""
        try:
            async with httpx.AsyncClient(base_url=self.api_base) as client:
                payload = {
                    "model": model,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                
                response = await client.post("/api/generate", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "No response")
                else:
                    return f"Error: {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def export_results(self) -> None:
        """Export results to CSV."""
        output = self.query_one("#output", RichLog)
        
        if not self.last_responses:
            output.write("[bold red]No results to export. Please run prompts first.[/]")
            return
            
        try:
            from datetime import datetime
            import csv
            import os
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_{timestamp}.csv"
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write headers
                writer.writerow(['Type', 'Content'])
                
                # Write prompts
                writer.writerow(['System Prompt', self.last_prompts['system']])
                writer.writerow(['User Prompt', self.last_prompts['user']])
                writer.writerow([])  # Empty row for separation
                
                # Write headers for responses
                writer.writerow(['Response Number', 'Response Content'])
                
                # Write responses
                for i, response in enumerate(self.last_responses, 1):
                    writer.writerow([f'Response {i}', response])
            
            output.write(f"[bold green]Results exported to {filename}[/]")
            
        except Exception as e:
            output.write(f"[bold red]Error exporting results: {str(e)}[/]")

if __name__ == "__main__":
    app = OllamaUI()
    app.run() 