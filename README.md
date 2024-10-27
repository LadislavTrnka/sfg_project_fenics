# (SFG) Mechanical response of elastic materials with density dependent Young modulus

In this repository, the behaviour of elastic bodies with _density dependent Young modulus_ is investigated in the infinitesimal strain regime (the concept of _density dependent Young modulus_ often occurs in the engineering literature on porous structures such as metal foams, aerogels or bones). Here, I numerically studied these models and compared their predictions to the predictions based on the classical models in linearised elasticity. Considered geometrical settings and problems are extension of a right circular cylinder, deflection of a thin plate, compression of a cube and bending of a rectangular prismatic beam. The problems are solved by finite element method using FEniCS library (legacy FEniCS 2019). For further details, see

Vít Průša and **Ladislav Trnka**. Mechanical response of elastic materials with density dependent Young modulus. _Applications in Engineering Science_, 14:100126, 2023. ISSN 2666-4968. [doi:10.1016/j.apples.2023.100126](https://doi.org/10.1016/j.apples.2023.100126)

{% include images.html url="fig/cylinder_n_1.png" description="Extension of a right circular cylinder, prediction by the standard linearised elasticity model (red), prediction by the model with density dependent Young modulus (yellow)" %}
