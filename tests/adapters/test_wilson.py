"""Module tests split from test_wilson_pipeline.py; shared numerical fixtures stay unchanged."""

from test_wilson_pipeline import (
    DenseWilsonBackend,
    _WilsonFixture,
    gamma,
    np,
    random_gauge,
    stencil,
    wilson_matrix,
)


class WilsonTests(_WilsonFixture):
    def test_independent_stencil_gamma5_and_gauge_covariance(self):
        sizes=(2,1,1,3)
        links=random_gauge(sizes)
        rng=np.random.default_rng(23)
        field=rng.normal(size=(3,1,1,2,4,3))+1j*rng.normal(size=(3,1,1,2,4,3))
        matrix=wilson_matrix(links,.4,t_boundary=-1)
        np.testing.assert_allclose((matrix@field.ravel()).reshape(field.shape),
                                   stencil(links,field,.4,-1),atol=2e-14)
        g5=np.kron(np.eye(6),np.kron(gamma(15),np.eye(3)))
        np.testing.assert_allclose(matrix.conj().T,g5@matrix@g5,atol=2e-14)
        omega=random_gauge(sizes,seed=25)[...,0,:,:]
        transformed=links.copy()
        for mu,axis in enumerate((3,2,1,0)):
            transformed[...,mu,:,:]=omega@links[...,mu,:,:]@np.roll(omega,-1,axis=axis).conj().swapaxes(-1,-2)
        field2=np.einsum("...ab,...sb->...sa",omega,field)
        expected=np.einsum("...ab,...sb->...sa",omega,stencil(links,field,.4,-1))
        np.testing.assert_allclose((wilson_matrix(transformed,.4)@field2.ravel()).reshape(field.shape),
                                   expected,atol=3e-14)

    def test_free_plane_wave_and_boundaries(self):
        shape=(3,2,2,3)
        links=np.broadcast_to(np.eye(3),(*shape,4,3,3)).astype(complex)
        t,z,y,x=np.indices(shape)
        spinor=np.arange(12).reshape(4,3)+.2j
        for boundary in (1,-1):
            p=np.array([2*np.pi/3,np.pi,0,(2*np.pi/3 if boundary==1 else np.pi/3)])
            wave=np.exp(1j*(p[0]*x+p[1]*y+p[2]*z+p[3]*t))
            field=wave[...,None,None]*spinor
            expected=(.4+sum(1-np.cos(p)))*spinor
            expected+=sum(1j*np.sin(p[mu])*(gamma(1<<mu)@spinor) for mu in range(4))
            result=(wilson_matrix(links,.4,t_boundary=boundary)@field.ravel()).reshape(field.shape)
            np.testing.assert_allclose(result,wave[...,None,None]*expected,atol=3e-14)
        with self.assertRaises(ValueError):
            wilson_matrix(links,.4,max_dimension=10)
        with self.assertRaises(ValueError):
            wilson_matrix(2*links,.4)
        rounded=random_gauge((2,1,1,3)).astype(np.complex64)
        self.assertEqual(wilson_matrix(rounded,.4).shape,(72,72))

    def test_backend_rejects_unsupported_physics_and_clears_previous_operator(self):
        links=random_gauge((2,1,1,3))
        parameters=dict(latt_size=(2,1,1,3),mass=.4,t_boundary=-1,xi_0=1,nu=1,
                        clover_coeff_t=0,clover_coeff_r=0,multigrid=None)
        backend=DenseWilsonBackend()
        backend.load(links,parameters)
        with self.assertRaises(NotImplementedError):
            backend.load(links,dict(parameters,clover_coeff_r=1))
        field=np.zeros((1,3,1,1,2,4,3),complex)
        with self.assertRaises(RuntimeError):
            backend.solve_into(field,field.copy(),tol=1e-11,maxiter=10)
