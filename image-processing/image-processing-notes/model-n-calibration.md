# Model n Calibration

## General camera model

There are 3 Coordinate Frames

* World {O}, Camera frame {C}, Image plane{Im}

The transformation between coordinate frames are

* Euclidean: {O} -> {C}, Mext : the camera extrinsic matrix
* Perspective Projection: {C}-> {Im}, Mint : the camera intrinsic matrix

![](<../../.gitbook/assets/image (290).png>)

![](<../../.gitbook/assets/image (293).png>)

![](<../../images/image (288).png>)

> x: Image Coordinates: (u,v,1)
>
> K: Intrinsic Matrix (3x3)
>
> R: Rotation (3x3)
>
> t: Translation (3x1)
>
> X: World Coordinates: (X,Y,Z,1)

### Extrinsic Matrix

Finding the camera external matrix of Mext, which is the transformation from {O} to {C}: Xc=\[R | T] Xo

> Here, R, T are from frame {C} to {O}. Depending on the notation, it can be the pose of {O} w.r.t {C}

![](<../../.gitbook/assets/image (285).png>)

### Intrinsic Matrix

#### Perspective projection in {C}: From {C} (3D) to {C} (2D)

> _p_ is NOT in pixel unit. It is in (mm) at distance 'f' from the {C} center point.

![](<../../.gitbook/assets/image (294).png>)

![](<../../.gitbook/assets/image (281).png>)

The relationship between P and _p_ are based on the similar triangle such as

![](<../../.gitbook/assets/image (283).png>)

#### Unit Conversion from {C} 2D (mm) to {Im} 2D

On the same image plane, the unit is changed from (mm) to (px). This depends on the mm-px scale unit, which is the image sensor pixel size.

> Here, we assume that there is NO skew and lens distortion

![](<../../.gitbook/assets/image (282).png>)

#### Intrinsic camera matrix, Mint

Putting the above two equations, the matrix Mint is the transformation between the camera frame {C} 3D(mm) and the image plane frame {Im} 2D(px)

![](<../../.gitbook/assets/image (284).png>)

> The scale factor cZ is not known from one frame of image. It is the actual distance of the object from the projection center.

Thus, from the image acquisition, we express the object position in px without knowing the exact scale as

![](<../../.gitbook/assets/image (291).png>)

## Camera Calibration

It is determining (1) Extrinsic Matrix (2) Intrinsic Matrix including lens distortion

![](<../../images/image (287).png>)

* Intrinsic Calibration
  * Lens distortion
  * Camera internal parameters
* Extrinsic Calibration
  * 6-DOF relative pose between the camera frame (3-D) and the world coordinate frame (3-D)
  * R, T are from {O} to {C}

### Intrinsic Calibration

Camera parameters

* focal length (mm)
* image center (px)
* effective pixel size (px/mm)

Lens Distortion

* Chromatic aberration
  * Index of refraction for glass varies as a function of wavelength.
  * Different color rays have different refraction
* Spherical aberration
  * Real lenses are not thin and suffers from geometric aberration
* Radial Distortion
  * Distortion at the periphery of the image

![](<../../.gitbook/assets/image (286).png>)

Xp: points’ location when lens is perfectly undistorted

Xd: points’ location when lens is distorted

Use a set of many points to find the distortion parameters such as corner points of a chess board.

![](<../../.gitbook/assets/image (280).png>)

### Zhang calibration method

Zhang, Zhengyou. "A flexible new technique for camera calibration." IEEE Transactions on pattern analysis and machine intelligence 22.11 (2000): 1330-1334.

Read here for detailed explanation

{% embed url="http://staff.fh-hagenberg.at/burger/publications/reports/2016Calibration/Burger-CameraCalibration-20160516.pdf" %}

## Appendix

#### Q What are affine, projection, projective, perspective, homography?

* Projection is mapping 3D to 2D
  * orthography, perspective(pinhole camera) and more
*   Transformation: 2D-2D or 3D-3D

    * Euclidean, affine, similarity, projective
    * Projective Transformation is also known as **perspective transformation or homography**

    \*\*\*\*

![](<../../images/image (301).png>)

#### Transformation: 2D-2D or 3D-3D (in the same dimension)

* projective: parallel lines converge to vanishing point

![](<../../.gitbook/assets/image (297).png>)

![](<../../images/image (298).png>)

Transformation Types

* Euclidean: preserves lengths and angles(isometry)
* Similarity: isotropic scaling preserves angle, shape, ratios of length, areas, angles
* Affine: non-singular transformation preserves parallelism A: n by n non-singular matrix (3x3 for 3D point)
* Projective: linear transformation on homogeneous n-vector (3D point: 4 x1 vector)
  * Perspective projection(3D-2D) is a subclass of projective transformation
  * P: non-singular nxn (4x4 for 3D point)

![](<../../.gitbook/assets/image (295).png>)

![](<../../images/image (303).png>)

![](<../../.gitbook/assets/image (296).png>)

![](<../../images/image (299).png>)

![](<../../images/image (302).png>)
