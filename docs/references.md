# References

The literature behind ZimaBlue. [`research.md`](research.md) is the argument —
which findings drove which decisions; this is the bibliography.

Every DOI here was resolved against the Crossref API on 2026-08-16, and author
lists, years, volumes and page ranges were taken from what came back rather
than from the citation as written. The arXiv-only entries were resolved the
same way against the arXiv API. Entries that could not be machine-verified
say so. Corrections made along the way are listed at the [end](#corrections).

## What the code actually implements

Most of this list is context. These are the papers you could open next to a
source file and follow line by line.

| Where | What | Reference |
|---|---|---|
| `dirt/types.py` | Grain settling velocity | Ferguson & Church (2004) |
| `estimation.py` | Zero-velocity updates for gyro bias | Foxlin (2005) |
| `estimation.py` | Odometry as a systematic error to calibrate, not noise | Borenstein & Feng (1996) |
| `controllers/systematic.py` | Nearest-frontier exploration | Yamauchi (1997) |
| `controllers/baseline.py` | Boustrophedon decomposition | Choset & Pignon (1997) |
| `metrics.py` | Coverage vs. removal as separate measures | Palleja et al. (2010); IEC 62929 |
| `sensors/base.py` | Noise, bias and drift terms of an inertial error model | IEEE Std 952-1997 |
| `sensors/base.py` | Why idealised sensors break transfer | Tobin et al. (2017) |
| `backends/` | Domain model as the API, engine as a strategy | Koenig & Howard (2004); Todorov et al. (2012) |
| `segment.py` | Promptable segmentation, and the multi-mask output the chooser ranks | Kirillov et al. (2023); Zhang et al. (2023) |
| `rl/env.py` | The environment interface | Towers et al. (2024) |

---

## Peer-reviewed articles

### Coverage path planning: foundations

1. Galceran, E., & Carreras, M. (2013). [A survey on coverage path planning for robotics](https://doi.org/10.1016/j.robot.2013.09.004). *Robotics and Autonomous Systems, 61*(12), 1258–1276.

2. Choset, H. (2001). [Coverage for robotics – A survey of recent results](https://doi.org/10.1023/A:1016639210559). *Annals of Mathematics and Artificial Intelligence, 31*(1–4), 113–126.

3. Choset, H., & Pignon, P. (1997). [Coverage path planning: The boustrophedon decomposition](https://publications.ri.cmu.edu/coverage-path-planning-the-boustrophedon-decomposition). *Proceedings of the 1st International Conference on Field and Service Robotics*, 216–222.

4. Acar, E. U., & Choset, H. (2002). [Sensor-based coverage of unknown environments: Incremental construction of Morse decompositions](https://doi.org/10.1177/027836402320556368). *The International Journal of Robotics Research, 21*(4), 345–366.

5. Gabriely, Y., & Rimon, E. (2001). [Spanning-tree based coverage of continuous areas by a mobile robot](https://doi.org/10.1023/A:1016610507833). *Annals of Mathematics and Artificial Intelligence, 31*(1–4), 77–98.

6. Yamauchi, B. (1997). [A frontier-based approach for autonomous exploration](https://doi.org/10.1109/CIRA.1997.613851). *Proceedings 1997 IEEE International Symposium on Computational Intelligence in Robotics and Automation*, 146–151.

7. Shen, Z., Gupta, S., Zhao, S., Zhou, D., Wang, G., Ren, Z., Ou, Y., Zhai, Y., & Chen, C. L. P. (2026). [Coverage path planning: Classical foundations, recent advances, and future directions](https://arxiv.org/abs/2607.10649). *arXiv*. https://doi.org/10.48550/arXiv.2607.10649

### Cleaning robots and coverage evaluation

8. Hofner, C., & Schmidt, G. (1995). [Path planning and guidance techniques for an autonomous mobile cleaning robot](https://doi.org/10.1016/0921-8890(94)00034-Y). *Robotics and Autonomous Systems, 14*(2–3), 199–212.

9. de Carvalho, R. N., Vidal, H. A., Vieira, P., & Ribeiro, M. I. (1997). [Complete coverage path planning and guidance for cleaning robots](https://doi.org/10.1109/ISIE.1997.649051). *Proceedings of the 1997 IEEE International Symposium on Industrial Electronics, 2*, 677–682.

10. Luo, C., & Yang, S. X. (2002). [A real-time cooperative sweeping strategy for multiple cleaning robots](https://doi.org/10.1109/ISIC.2002.1157841). *Proceedings of the IEEE International Symposium on Intelligent Control*, 660–665.

11. Oh, J. S., Choi, Y. H., Park, J. B., & Zheng, Y. F. (2004). [Complete coverage navigation of cleaning robots using triangular-cell-based map](https://doi.org/10.1109/TIE.2004.825197). *IEEE Transactions on Industrial Electronics, 51*(3), 718–726.

12. Palacin, J., Salse, J. A., Valganon, I., & Clua, X. (2004). [Building a mobile robot for a floor-cleaning operation in domestic environments](https://doi.org/10.1109/TIM.2004.834093). *IEEE Transactions on Instrumentation and Measurement, 53*(5), 1418–1424.

13. Palacin, J., Palleja, T., Valganon, I., Pernia, R., & Roca, J. (2005). [Measuring coverage performances of a floor cleaning mobile robot using a vision system](https://doi.org/10.1109/ROBOT.2005.1570771). *Proceedings of the 2005 IEEE International Conference on Robotics and Automation*, 4236–4241.

14. Palleja, T., Tresanchez, M., Teixido, M., & Palacin, J. (2010). [Modeling floor-cleaning coverage performances of some domestic mobile robots in a reduced scenario](https://doi.org/10.1016/j.robot.2009.07.030). *Robotics and Autonomous Systems, 58*(1), 37–45.

15. Lee, T.-K., Baek, S., & Oh, S.-Y. (2011). [Sector-based maximal online coverage of unknown environments for cleaning robots with limited sensing](https://doi.org/10.1016/j.robot.2011.05.005). *Robotics and Autonomous Systems, 59*(10), 698–710.

16. Batista, V. R., & Zampirolli, F. A. (2019). [Optimising robotic pool-cleaning with a genetic algorithm](https://doi.org/10.1007/s10846-018-0953-y). *Journal of Intelligent & Robotic Systems, 95*(2), 443–458.

### Underwater coverage and localization

17. Englot, B., & Hover, F. S. (2013). [Three-dimensional coverage planning for an underwater inspection robot](https://doi.org/10.1177/0278364913490046). *The International Journal of Robotics Research, 32*(9–10), 1048–1073.

18. Galceran, E., Campos, R., Palomeras, N., Ribas, D., Carreras, M., & Ridao, P. (2015). [Coverage path planning with real-time replanning and surface reconstruction for inspection of three-dimensional underwater structures using autonomous underwater vehicles](https://doi.org/10.1002/rob.21554). *Journal of Field Robotics, 32*(7), 952–983.

19. Yan, M., Zhu, D., & Yang, S. X. (2012). [Complete coverage path planning in an unknown underwater environment based on D-S data fusion real-time map building](https://doi.org/10.1155/2012/567959). *International Journal of Distributed Sensor Networks, 8*(10), 567959.

20. Morin, M., Abi-Zeid, I., Petillot, Y., & Quimper, C.-G. (2013). [A hybrid algorithm for coverage path planning with imperfect sensors](https://doi.org/10.1109/IROS.2013.6697225). *2013 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 5988–5993.

21. Ferrera, M., Creuze, V., Moras, J., & Trouvé-Peloux, P. (2019). [AQUALOC: An underwater dataset for visual–inertial–pressure localization](https://doi.org/10.1177/0278364919883346). *The International Journal of Robotics Research, 38*(14), 1549–1559.

22. Rahman, S., Quattrini Li, A., & Rekleitis, I. (2022). [SVIn2: A multi-sensor fusion-based underwater SLAM system](https://doi.org/10.1177/02783649221110259). *The International Journal of Robotics Research, 41*(11–12), 1022–1042.

23. Yan, L., Chang, S., Wang, X., Zhang, L., & Liu, J. (2024). [A dual-stage coverage path planning method for bathymetric survey using an AUV in graph-based SLAM framework considering positioning uncertainty](https://doi.org/10.1016/j.oceaneng.2024.119252). *Ocean Engineering, 312*, 119252.

24. Ibrahim, A., Rego, F. F. C., & Busvelle, É. (2025). [Comparison of innovative strategies for the coverage problem: Path planning, search optimization, and applications in underwater robotics](https://doi.org/10.3390/jmse13071369). *Journal of Marine Science and Engineering, 13*(7), 1369. Preprint: [arXiv:2506.15376](https://arxiv.org/abs/2506.15376).

25. Bucci, A., & Ridolfi, A. (2026). [Multi-session perception-aware coverage path planning for active semantic SLAM and automatic change detection](https://doi.org/10.1016/j.oceaneng.2026.125170). *Ocean Engineering, 355*, 125170.

26. Amer, A., Mehindratta, M., Brodskiy, Y., Wehbe, B., & Kayacan, E. (2025). [REACT: Real-time entanglement-aware coverage path planning for tethered underwater vehicles](https://arxiv.org/abs/2507.10204). *arXiv*. https://doi.org/10.48550/arXiv.2507.10204

### Sensing, estimation and odometry

27. Borenstein, J., & Feng, L. (1996). [Measurement and correction of systematic odometry errors in mobile robots](https://doi.org/10.1109/70.544770). *IEEE Transactions on Robotics and Automation, 12*(6), 869–880.

28. Foxlin, E. (2005). [Pedestrian tracking with shoe-mounted inertial sensors](https://doi.org/10.1109/MCG.2005.140). *IEEE Computer Graphics and Applications, 25*(6), 38–46.

29. Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). [Domain randomization for transferring deep neural networks from simulation to the real world](https://doi.org/10.1109/IROS.2017.8202133). *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 23–30.

### Simulation

30. Koenig, N., & Howard, A. (2004). [Design and use paradigms for Gazebo, an open-source multi-robot simulator](https://doi.org/10.1109/IROS.2004.1389727). *2004 IEEE/RSJ International Conference on Intelligent Robots and Systems, 3*, 2149–2154.

31. Todorov, E., Erez, T., & Tassa, Y. (2012). [MuJoCo: A physics engine for model-based control](https://doi.org/10.1109/IROS.2012.6386109). *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 5026–5033.

### Segmentation and learned control

These are the two optional extras. Both are used as tools rather than
reimplemented: ZimaBlue runs a published SAM checkpoint and implements a
Gymnasium interface, and there is no algorithm from either paper in the source.

32. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). [Segment Anything](https://arxiv.org/abs/2304.02643). *arXiv:2304.02643*. The promptable-segmentation model behind `SamSegmenter`, and the source of the multi-mask output the chooser ranks.

33. Zhang, C., Han, D., Qiao, Y., Kim, J. U., Bae, S.-H., Lee, S., & Hong, C. S. (2023). [Faster Segment Anything: Towards Lightweight SAM for Mobile Applications](https://arxiv.org/abs/2306.14289). *arXiv:2306.14289*. MobileSAM — the 45 MB distillation that makes running this on a CPU reasonable.

34. Towers, M., Kwiatkowski, A., Terry, J., Balis, J. U., De Cola, G., Deleu, T., Goulão, M., Kallinteris, A., Krimmel, M., KG, A., Perez-Vicente, R., Pierré, A., Schulhoff, S., Tai, J. J., Tan, H., & Younis, O. G. (2024). [Gymnasium: A Standard Interface for Reinforcement Learning Environments](https://arxiv.org/abs/2407.17032). *arXiv:2407.17032*. The interface `PoolCleaningEnv` implements.

35. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). *arXiv:1707.06347*. Not implemented here — cited because it is what the throughput estimates in [`ml.md`](ml.md) assume you will point at the env.

### Sediment and settling

36. Ferguson, R. I., & Church, M. (2004). [A simple universal equation for grain settling velocity](https://doi.org/10.1306/051204740933). *Journal of Sedimentary Research, 74*(6), 933–937.

37. Mendrik, F., Fernández, R., Hackney, C. R., Waller, C., & Parsons, D. R. (2023). [Non-buoyant microplastic settling velocity varies with biofilm growth and ambient water salinity](https://doi.org/10.1038/s43247-023-00690-z). *Communications Earth & Environment, 4*(1), 30.

## Standards

1. IEC. (2014). [IEC 62929:2014 — Cleaning robots for household use: Dry cleaning — Methods of measuring performance](https://webstore.iec.ch/en/publication/7477). International Electrotechnical Commission.

2. IEEE. (1998). [IEEE Std 952-1997 — Standard specification format guide and test procedure for single-axis interferometric fiber optic gyros](https://doi.org/10.1109/IEEESTD.1998.86153). IEEE. *(Annex C is the canonical Allan-variance treatment of inertial sensor noise.)*

## Books

1. Choset, H., Lynch, K. M., Hutchinson, S., Kantor, G. A., Burgard, W., Kavraki, L. E., & Thrun, S. (2005). [*Principles of robot motion: Theory, algorithms, and implementations*](https://mitpress.mit.edu/9780262033275/principles-of-robot-motion/). MIT Press.

2. LaValle, S. M. (2006). [*Planning algorithms*](https://doi.org/10.1017/CBO9780511546877). Cambridge University Press. Free online edition: <https://msl.cs.illinois.edu/~lavalle/planning/>.

3. Thrun, S., Burgard, W., & Fox, D. (2005). [*Probabilistic robotics*](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/). MIT Press.

4. Correll, N., Hayes, B., Heckman, C., & Roncone, A. (2022). [*Introduction to autonomous robots: Mechanisms, sensors, actuators, and algorithms*](https://mitpress.mit.edu/9780262047555/introduction-to-autonomous-robots/). MIT Press.

## Dissertations and theses

1. Karapetyan, N. (2021). [*Robot area coverage path planning in aquatic environments*](https://scholarcommons.sc.edu/etd/6730/) [Doctoral dissertation, University of South Carolina].

## Patents

Neither entry below could be verified from CI: Google Patents and Espacenet
both block automated access, and a web search did not surface them. They are
recorded as supplied. Check them at
<https://worldwide.espacenet.com/> before citing either in anything that
matters.

1. Wei, J., Guo, Y., Shang, C., Ma, Y., & He, J. (2024). [Path planning method and apparatus for swimming pool cleaning robot](https://patents.google.com/patent/EP4390602A1/en) (European Patent Application No. EP4390602A1). *Unverified.*

2. [Path planning method and device of swimming pool cleaning robot](https://patents.google.com/patent/CN114895691B/en) (Chinese Patent No. CN114895691B). *Unverified.*

## Technical resources

1. Foxglove. [*MCAP specification*](https://mcap.dev/specification/index.html). The default `rosbag2` storage format from ROS 2 Iron onward; the design ZimaBlue's `.zbr` container borrows from and deliberately diverges from.

2. Hurliman, J. (2021). [*Evaluation of robotics data recording file formats*](https://mcap.dev/files/evaluation.pdf). Foxglove.

3. NVIDIA. [*Isaac Lab: Reproducibility and determinism*](https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html). The clearest public statement of why GPU simulation is not bit-reproducible across hardware.

4. NVIDIA. [*Using OpenUSD for modular and scalable robotic simulation*](https://developer.nvidia.com/blog/using-openusd-for-modular-and-scalable-robotic-simulation-and-development/).

5. Furgale, P., et al. [*IMU noise model*](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model). Kalibr wiki. The practical white-noise-plus-random-walk model ZimaBlue's sensor pipeline implements.

6. LaValle, S. M. (2006). [*Planning algorithms*](https://msl.cs.illinois.edu/~lavalle/planning/) [Online edition]. University of Illinois.

## Corrections

Errors found while verifying the list this file grew from. Recorded rather than
quietly fixed, because anyone who copied the original will want to know.

- **Palacin et al., "Measuring coverage performances of a mobile robot for
  floor-cleaning applications", *Autonomous Robots* 18, 97–111 (2004)** does not
  exist. The DOI `10.1023/B:AURO.0000016865.23815.2f` returns 404 and Crossref
  has no such paper in that journal or volume. The nearest real works by those
  authors are entries 12 and 13 above — an ICRA 2005 paper with an almost
  identical title, and a 2004 IEEE TIM paper — and both are now cited directly.

- **Yan, Zhu & Yang (2012)** was attributed to the *International Journal of
  Advanced Robotic Systems*. It appeared in the *International Journal of
  Distributed Sensor Networks*, 8(10).

- **Palleja et al. (2010)** was credited to "Pallejà, Palacín, Valgañón, Pernía
  & Roca". The actual authors are Palleja, Tresanchez, Teixido and Palacin.

- **Lee, Baek & Oh (2011)** had no authors listed at all.

- **Galceran et al. (2015)** is volume 32, *issue 7* — the online-first version
  dates from 2014, which is why some indexes disagree on the year.

- **Luo & Yang (2002)** was linked to ResearchGate; replaced with the DOI.

- **Ibrahim et al. (2025)** appeared twice, once as a journal article and once
  as its own preprint. Merged, with the preprint noted on the journal entry.
