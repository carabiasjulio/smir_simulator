from dataclasses import dataclass
import numpy as np
from typing import Union
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import mat73

@dataclass
class Hom:
    isRigid: bool
    r: float
    el: np.ndarray  # (M,)
    az: np.ndarray  # (M,)
    cart: np.ndarray  # (M, 3)
    dist: np.ndarray  # (M, M)
    w: Union[np.ndarray, int]
    gain: np.ndarray  # (M,)
    emReference: Union[np.ndarray, None] = None  # indices en el Eigenmike original


def _s2c(el: np.ndarray, az: np.ndarray, r: float) -> np.ndarray:
    """
    Coord. esféricas (el, az, r) -> cartesianas (x,y,z)
    el: elevación [0..pi] desde eje +z
    az: azimut [0..2pi] antihorario desde +x
    """
    el = np.asarray(el).reshape(-1)
    az = np.asarray(az).reshape(-1)
    x = r * np.sin(el) * np.cos(az)
    y = r * np.sin(el) * np.sin(az)
    z = r * np.cos(el)
    return np.column_stack([x, y, z])


def get_eigenmike(r, az, el, gain, w, isRigid) -> Hom:
    """
    Versión mínima del MATLAB SHTools.getEigenmike().
    Sustituye estos arrays por los tuyos si ya los tienes cargados.
    """
    cart = _s2c(el, az, r)
    # Matriz de distancias entre mics
    dist = squareform(pdist(cart))

    return Hom(isRigid=isRigid, r=r, el=el, az=az, cart=cart, dist=dist, w=w, gain=gain)


def get_grid(Q: int, mat_path: str = "HOAGridData.mat") -> dict:
    """
    Carga HOAGridData.mat (v7.3, HDF5) y devuelve el grid N correspondiente
    con las claves 'cart', 'w' y 'N'.

    Parámetros
    ----------
    Q : int
        Número de puntos del grid (por ejemplo, 32).
    mat_path : str
        Ruta del archivo HOAGridData.mat (v7.3 HDF5).

    Devuelve
    --------
    dict con:
        - 'cart': ndarray (M, 3)
        - 'w'   : ndarray (M,)
        - 'N'   : int  -> ceil(sqrt(Q) - 1)
    """
    N = int(np.ceil(np.sqrt(Q) - 1))
    if N > 29:
        raise ValueError("Upto 900 nodes are supported (N must be <= 29).")

    data = mat73.loadmat(mat_path)  # dict plano
    HOA = data.get("HOAGrid", None)
    if HOA is None:
        raise KeyError("Variable 'HOAGrid' no encontrada en el .mat (mat73).")
        
    return HOA[N - 1][0]


def reduced_eigenmike(Q: int, hom: Union[Hom, None] = None, grid: Union[dict, None] = None) -> Hom:
    """
    Replica SHTools.reducedEigenmike(Q, hom) de MATLAB.

    Selecciona Q micrófonos del Eigenmike (hom) eligiendo, para cada nodo del
    grid HOA (grid['cart']), el micrófono más cercano en distancia euclídea.

    Parámetros
    ----------
    Q : int
        Nº de micrófonos deseados.
    hom : Hom | None
        Estructura completa del Eigenmike. Si None, se usa get_eigenmike().
    grid : dict | None
        Diccionario con claves:
          - 'cart': (Q, 3) posiciones cartesianas de los nodos HOA
          - 'w'   : (Q,) pesos del grid
        Si None, se usa HOAGrid.get_grid(Q) (stub aquí).

    Devuelve
    --------
    Hom
        Nueva estructura `Hom` reducida a Q micrófonos, con:
        - el, az, cart, dist, gain submuestreados
        - w = grid['w']
        - emReference = índices (en el Eigenmike original) de los mics elegidos
    """
    if hom is None:
        hom = get_eigenmike()

    if grid is None:
        grid = get_grid(Q)

    a = np.asarray(hom.cart)  # (M, 3)
    b = np.asarray(grid["cart"])  # (Q, 3)

    # Distancias: cada nodo del grid b a todos los mics a -> índice del más cercano
    D = distance_matrix(b, a)  # (Q, M)

    ind = np.argmin(D, axis=1)  # (Q,)

    # Submuestreo
    el_sub = hom.el[ind]
    az_sub = hom.az[ind]
    cart_sub = hom.cart[ind, :]
    gain_sub = hom.gain[ind]

    # Matriz de distancias nueva (Q x Q)
    dist_sub = squareform(pdist(cart_sub))
    
    # Construir hom reducido
    hom_reduced = Hom(
        isRigid=hom.isRigid,
        r=hom.r,
        el=el_sub,
        az=az_sub,
        cart=cart_sub,
        dist=dist_sub,
        w=np.asarray(grid["w"]),
        gain=gain_sub,
        emReference=ind.astype(int),
    )

    # Nota: Igual que en MATLAB, no se fuerza unicidad en 'ind':
    # puede haber micrófonos repetidos si dos nodos del grid
    # caen más cerca del mismo mic, tal y como hace pdist2+min en el script original.
    return hom_reduced