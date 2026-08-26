# Differentiable drive physics

The smooth core of a differential drive has exact derivatives. ZimaBlue
exposes them without requiring JAX, PyTorch or a second dynamics model:

```python
import numpy as np
import zimablue as zb

state = np.array([1.0, 2.0, 0.3])       # x, y, heading
command = np.array([0.20, 0.27])        # left/right ground speed, m/s

next_state, jacobian = zb.differentiable_drive_step(
    state,
    command,
    dt=0.02,
    track_width=0.38,
)

print(jacobian.state.shape)       # (3, 3)
print(jacobian.control.shape)     # (3, 2)
print(jacobian.track_width.shape) # (3,)
```

The integrator is the same constant-twist exact arc used by the fast backend,
written as

$$
\Delta p = v\,\Delta t\,\operatorname{sinc}(z)
\begin{bmatrix}\cos(\theta+z)\\\sin(\theta+z)\end{bmatrix},
\qquad z = \frac{\omega\Delta t}{2}.
$$

A series expansion evaluates `sinc` and its derivative near zero. Straight
motion therefore has a finite, correct turn-rate derivative instead of falling
through a numerical `if omega == 0` branch.

## Rollouts

`differentiable_rollout()` returns every pose, every local Jacobian and the
terminal pose derivative with respect to every wheel command:

```python
trajectory = zb.differentiable_rollout(
    state,
    commands,             # shape (steps, 2)
    dt=0.02,
    track_width=0.38,
)
gradient = trajectory.terminal_control_jacobian  # (3, steps, 2)
```

The rollout propagates derivatives with the chain rule. This is the primitive
needed by shooting-method MPC, command optimisation and local sensitivity
analysis. Analytical tests compare state, command, track-width and multi-step
Jacobians against central finite differences.

## Domain boundary

The controls are **ground wheel speeds after drivetrain limits and slip**.
These derivatives cover free-space rigid motion. They deliberately do not
differentiate:

- speed or acceleration clipping;
- the clipped empirical slip law;
- wall or robot contact;
- stuck-state transitions;
- dirt capture and filter saturation.

Those operations are piecewise or discrete. Returning a smooth-looking number
through a collision would be less useful than stating the boundary. Use the
full `Simulation` to verify any optimised command sequence, and reject or
replan trajectories that enter contact or actuator saturation.