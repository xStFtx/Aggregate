import sympy as sp
import numpy as np
from scipy.linalg import eig

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        norm_val = self.norm()
        return Quaternion(self.w/norm_val, self.x/norm_val, self.y/norm_val, self.z/norm_val)

    def inverse(self):
        norm_sq = self.norm()**2
        conj = self.conjugate()
        return Quaternion(conj.w/norm_sq, conj.x/norm_sq, conj.y/norm_sq, conj.z/norm_sq)

    def __truediv__(self, other):
        return self * other.inverse()

    def dot(self, other):
        return self.w*other.w + self.x*other.x + self.y*other.y + self.z*other.z

    def cross(self, other):
        x = self.y*other.z - self.z*other.y
        y = self.z*other.x - self.x*other.z
        z = self.x*other.y - self.y*other.x
        return Quaternion(0, x, y, z)

    def to_angle_axis(self):
        angle = 2 * np.arccos(self.w)
        sin_theta = np.sqrt(1 - self.w**2)
        x = self.x / sin_theta
        y = self.y / sin_theta
        z = self.z / sin_theta
        return angle, (x, y, z)

    def exponential(self):
        A = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        exp_w = np.exp(self.w)
        sA = np.sin(A) / A if A != 0 else 1
        return Quaternion(exp_w * np.cos(A), exp_w * sA * self.x, exp_w * sA * self.y, exp_w * sA * self.z)

    def logarithm(self):
        A = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        norm_q = self.norm()
        if A == 0:
            return Quaternion(np.log(norm_q), 0, 0, 0)
        t = np.arccos(self.w / norm_q)
        factor = t / A
        return Quaternion(np.log(norm_q), factor * self.x, factor * self.y, factor * self.z)

    def __mul__(self, other):
        w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
        x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
        y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
        z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        return Quaternion(w, x, y, z)

    def to_matrix(self):
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [ w, -x, -y, -z],
            [ x,  w, -z,  y],
            [ y,  z,  w, -x],
            [ z, -y,  x,  w]
        ])

    def __repr__(self):
        return f"({self.w}, {self.x}, {self.y}, {self.z})"

def inner_product(v1, v2):
    return sum([a.conjugate()*b for a, b in zip(v1, v2)])

def fourier_transform(signal):
    return np.fft.fft(signal)

def inverse_fourier_transform(signal):
    return np.fft.ifft(signal)

def laplace_transform(f):
    t, s = sp.symbols('t s')
    return sp.laplace_transform(f, t, s)

def inverse_laplace_transform(F):
    t, s = sp.symbols('t s')
    return sp.inverse_laplace_transform(F, s, t)

def convolution(f, g, t):
    tau = sp.symbols('tau')
    return sp.integrate(f.subs(t, tau) * g.subs(t, t - tau), (tau, -sp.oo, sp.oo))

def quaternion_eigen(q):
    matrix = q.to_matrix()
    eigenvalues, eigenvectors = eig(matrix)
    return eigenvalues, eigenvectors

def slerp(q1, q2, t):
    q1 = q1.normalize()
    q2 = q2.normalize()
    dot_product = q1.dot(q2)

    if dot_product < 0:
        q2 = -q2
        dot_product = -dot_product

    if dot_product > 0.95:
        result = q1 + t*(q2 - q1)
        return result.normalize()

    theta_0 = np.arccos(dot_product)
    theta = theta_0 * t
    q3 = (q2 - q1*dot_product).normalize()

    return q1*np.cos(theta) + q3*np.sin(theta)

if __name__ == "__main__":
    q1 = Quaternion(1, 2, 3, 4)
    q2 = Quaternion(2, 3, 4, 5)
    print(q1 * q2)

    v1 = [1+2j, 2+3j, 3+4j]
    v2 = [4+5j, 5+6j, 6+7j]
    print(inner_product(v1, v2))

    signal = [0, 1, 0, -1]
    print(fourier_transform(signal))

    t = sp.symbols('t')
    f = sp.exp(-t)
    print(laplace_transform(f))

    f = sp.Heaviside(t)
    g = sp.exp(-t) * sp.Heaviside(t)
    print(convolution(f, g, t))

    q = Quaternion(1, 2, 3, 4)
    eigenvalues, eigenvectors = quaternion_eigen(q)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:", eigenvectors)

    q3 = slerp(q1, q2, 0.5)
    print("Slerp Interpolation:", q3)
