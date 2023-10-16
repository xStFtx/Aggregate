from qmath import Quaternion
import nn

def demonstrate_quaternion_operations():
    q1 = Quaternion(1, 2, 3, 4)
    q2 = Quaternion(2, 3, 4, 5)
    print(f"q1: {q1}")
    print(f"q2: {q2}")
    
    # Some basic operations for demonstration
    print(f"q1 * q2: {q1 * q2}")
    print(f"q1 + q2: {q1 + q2}")
    angle, axis = q1.to_angle_axis()
    print(f"Angle-axis representation of q1: Angle = {angle}, Axis = {axis}")
    
def train_rl_model():
    nn.train_dqn()

if __name__ == "__main__":
    print("Demonstrating Quaternion Operations:")
    demonstrate_quaternion_operations()
    print("\nTraining RL Model using Quaternion Neural Network:")
    train_rl_model()
