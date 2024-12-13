"""Entrance."""
import env
import os
import imageio
import numpy as np

if __name__ == '__main__':
    environment, scheduler = env.load()
    frames = []  # Initialize empty frame list
    
    while not environment.terminated():
        frame = environment._render_()
        # Convert frame to uint8 and ensure consistent size
        frame = frame.astype(np.uint8)
        frames.append(frame)
        actions = scheduler.schedule()
        
        # Store action in file
        with open('__cache__/action.txt', 'a') as f:
            f.write(str(actions) + '\n')

    print('END')

    # Save as GIF with optimization
    if not os.path.exists('__cache__/animation'):
        os.makedirs('__cache__/animation')
    
    # Save with reduced size and optimizations
    imageio.mimsave(
        '__cache__/animation/schedule.gif',
        frames,
        fps=2,
        optimize=True,
        subrectangles=True
    )