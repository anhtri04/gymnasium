"""Read and display TensorBoard event files"""
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd

def read_tensorboard_log(log_path):
    """Read a TensorBoard event file and return metrics as a DataFrame"""
    data = []
    
    for event in summary_iterator(log_path):
        step = event.step
        wall_time = event.wall_time
        
        for value in event.summary.value:
            data.append({
                'step': step,
                'wall_time': wall_time,
                'tag': value.tag,
                'value': value.simple_value
            })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    log_file = "tensorboard_logs/PPO_4/events.out.tfevents.1771872967.LAPTOP-EBR1E3GQ.20468.0"
    
    print(f"Reading: {log_file}\n")
    df = read_tensorboard_log(log_file)
    
    # Display summary
    print("Available metrics:")
    print(df['tag'].unique())
    print(f"\nTotal events: {len(df)}")
    print(f"\nSteps range: {df['step'].min()} to {df['step'].max()}")
    
    # Show sample data
    print("\nSample data:")
    print(df.head(20))
    
    # Show statistics for key metrics
    print("\n=== Training Statistics ===")
    for tag in df['tag'].unique():
        tag_data = df[df['tag'] == tag]['value']
        print(f"\n{tag}:")
        print(f"  Mean: {tag_data.mean():.4f}")
        print(f"  Min: {tag_data.min():.4f}")
        print(f"  Max: {tag_data.max():.4f}")
        print(f"  Final: {tag_data.iloc[-1]:.4f}")
