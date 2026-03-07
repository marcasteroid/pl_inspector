import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pl_inspector import trace_training

# ----------------------------------------------------------------------
# 1. Define a Tiny Toy Model
# ----------------------------------------------------------------------
# We create a simple linear model to learn a basic mapping.
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        
        # Initialize weights to something non-zero for immediate learning
        with torch.no_grad():
            self.linear.weight.fill_(2.0)
            self.linear.bias.fill_(0.5)

    def forward(self, x):
        return self.linear(x)

def main():
    print("Starting Torch training diagnostic demo...\n")
    
    model = ToyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    # Toy data: learning the function y = 3x + 1
    x_train = torch.tensor([[1.0], [2.0], [3.0]])
    y_train = torch.tensor([[4.0], [7.0], [10.0]])

    # ------------------------------------------------------------------
    # 2. Create and Attach Training Diagnostics Session
    # ------------------------------------------------------------------
    # The `trace_training` entry point provides a persistent session that 
    # automatically stores metrics and metadata to the specified directory.
    print("Initializing training session...")
    session = trace_training(interface="torch", save_dir="./demo_runs")
    
    # Attach the PyTorch model so the session can inspect gradient and 
    # parameter norms during the training loop.
    session.attach(model)
    print(f"Session started! Run ID: {session.run_id}")

    # ------------------------------------------------------------------
    # 3. Running the Training Loop
    # ------------------------------------------------------------------
    num_epochs = 20
    print(f"\nTraining for {num_epochs} steps...")
    
    for step in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(x_train)
        loss = loss_fn(predictions, y_train)
        
        # Backward pass
        loss.backward()
        
        # Before taking an optimization step, we log the diagnostics.
        # This records the global gradient norm (now populated) and 
        # computed parameter updates, and saves it seamlessly.
        session.log_step(loss=loss.item(), step=step)
        
        # Parameter update
        optimizer.step()
        
        if step % 5 == 0 or step == num_epochs - 1:
            print(f"  Step {step:2d} | Loss: {loss.item():.4f}")

    # ------------------------------------------------------------------
    # 4. Plotting and Exporting
    # ------------------------------------------------------------------
    # The session provides high-level plotting utilities wrapping `matplotlib`.
    print("\nTraining complete. Generating plots...")
    fig, (ax_loss, ax_grad) = plt.subplots(1, 2, figsize=(12, 4))
    
    session.plot_loss(ax=ax_loss)
    session.plot_gradient_norms(ax=ax_grad)
    
    plt.tight_layout()
    plot_path = session.storage.run_dir / "training_trends.png"
    plt.savefig(plot_path)
    print(f"Saved plots to: {plot_path}")

    # Export the final metadata and summary stats.
    # This also persists `summary.json` so tools can easily discover this run.
    run_dir = session.export_json()
    print(f"\nRun successfully exported!\nData directory: {run_dir}")


if __name__ == "__main__":
    main()
