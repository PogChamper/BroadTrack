// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
// 
// This is a license agreement ("Agreement") between you (called "Licensee" or "You" in this Agreement) and EVS Broadcast Equipment SA. (called "Licensor" in this Agreement).  All rights not specifically granted to you in this Agreement are reserved for Licensor.
// 
// RESERVATION OF OWNERSHIP AND GRANT OF LICENSE:
// Licensor retains exclusive ownership of any copy of the Software (as defined below) licensed under this Agreement and hereby grants to Licensee a personal, non-exclusive, non-transferable license to use the Software for noncommercial research purposes, without the right to sublicense, pursuant to the terms and conditions of this Agreement.  As used in this Agreement, the term "Software" means (i) the actual copy of all or any portion of code for program routines made accessible to Licensee by Licensor pursuant to this Agreement, inclusive of backups, updates, and/or merged copies permitted hereunder or subsequently supplied by Licensor,  including all or any file structures, programming instructions, user interfaces and screen formats and sequences as well as any and all documentation and instructions related to it, and (ii) all or any derivatives and/or modifications created or made by You to any of the items specified in (i).
// CONFIDENTIALITY: Licensee acknowledges that the Software is proprietary to Licensor, and as such, Licensee agrees to receive all such materials in confidence and use the Software only in accordance with the terms of this Agreement.  Licensee agrees to use reasonable effort to protect the Software from unauthorized use, reproduction, distribution, or publication.
// COPYRIGHT: The Software is owned by Licensor and is protected by copyright laws and applicable international treaties and/or conventions.
// PERMITTED USES:  The Software may be used for your own noncommercial internal research purposes. You understand and agree that Licensor is not obligated to implement any suggestions and/or feedback you might provide regarding the Software, but to the extent Licensor does so, you are not entitled to any compensation related thereto.
// DERIVATIVES: You may create derivatives of or make modifications to the Software, however, You agree that all and any such derivatives and modifications will be owned by Licensor and become a part of the Software licensed to You under this Agreement.  You may only use such derivatives and modifications for your own noncommercial internal research purposes, and you may not otherwise use, distribute or copy such derivatives and modifications in violation of this Agreement.
// BACKUPS:  If Licensee is an organization, it may make that number of copies of the Software necessary for internal noncommercial use at a single site within its organization provided that all information appearing in or on the original labels, including the copyright and trademark notices are copied onto the labels of the copies.
// USES NOT PERMITTED:  You may not distribute, copy or use the Software except as explicitly permitted herein. Licensee has not been granted any trademark license as part of this Agreement and may not use the name or mark "EVS" or any renditions thereof without the prior written permission of Licensor.
// You may not sell, rent, lease, sublicense, lend, time-share or transfer, in whole or in part, or provide third parties access to prior or present versions (or any parts thereof) of the Software.
// ASSIGNMENT: You may not assign this Agreement or your rights hereunder without the prior written consent of Licensor. Any attempted assignment without such consent shall be null and void.
// TERM: The term of the license granted by this Agreement is from Licensee's acceptance of this Agreement by downloading the Software or by using the Software until terminated as provided below.
// The Agreement automatically terminates without notice if you fail to comply with any provision of this Agreement.  Licensee may terminate this Agreement by ceasing using the Software.  Upon any termination of this Agreement, Licensee will delete any and all copies of the Software. You agree that all provisions which operate to protect the proprietary rights of Licensor shall remain in force should breach occur and that the obligation of confidentiality described in this Agreement is binding in perpetuity and, as such, survives the term of the Agreement.
// FEE: Provided Licensee abides completely by the terms and conditions of this Agreement, there is no fee due to Licensor for Licensee's use of the Software in accordance with this Agreement.
// DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON-INFRINGEMENT.  LICENSEE BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND RELATED MATERIALS.
// SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is provided as part of this Agreement. 
// EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent permitted under applicable law, Licensor shall not be liable for direct, indirect, special, incidental, or consequential damages or lost profits related to Licensee's use of and/or inability to use the Software, even if Licensor is advised of the possibility of such damage.
// EXPORT REGULATION: Licensee agrees to comply with any and all applicable export control laws, regulations, and/or other laws related to embargoes and sanction programs.
// SEVERABILITY: If any provision(s) of this Agreement shall be held to be invalid, illegal, or unenforceable by a court or other tribunal of competent jurisdiction, the validity, legality and enforceability of the remaining provisions shall not in any way be affected or impaired thereby.
// NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right or remedy under this Agreement shall be construed as a waiver of any future or other exercise of such right or remedy by Licensor.
// GOVERNING LAW: This Agreement shall be construed and enforced in accordance with the laws of Belgium without reference to conflict of laws principles.  You consent to the exclusive jurisdiction of the courts of Liège.
// ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and entire agreement between Licensee and Licensor as to the matter set forth herein and supersedes any previous agreements, understandings, and arrangements between the parties relating hereto.


#include "core.h"
#include "Camera.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>

struct FixedPointReprojectionError
{
    FixedPointReprojectionError(const Eigen::Vector3<double> &fixedWorldPoint,
                                const Eigen::Vector2<double> &observedImagePoint)
        : _fixedWorldPoint(fixedWorldPoint),
          _observedImagePoint(observedImagePoint) {}

    template <typename T>
    bool operator()(const T *const camera_p,
                    T *residuals) const
    {
        const Eigen::Vector3<T> angleAxis(&camera_p[0]);
        const Eigen::Vector3<T> position(&camera_p[3]);
        const T focalLength = camera_p[6];
        const T k1 = camera_p[7];

        const Eigen::Vector3<T> worldPoint = _fixedWorldPoint.cast<T>() - position;

        Eigen::Vector3<T> cameraPoint = worldPoint;
        ceres::AngleAxisRotatePoint(angleAxis.data(), worldPoint.data(), cameraPoint.data());
        Eigen::Vector2<T> normalizedPlaneProjection = cameraPoint.hnormalized();
        T r2 = normalizedPlaneProjection.squaredNorm(); // normalizedPlaneProjectionRadiusSquared
        T distortion = 1.0 + k1 * r2;
        Eigen::Vector2<T> predictedImagePoint = focalLength * distortion * normalizedPlaneProjection;
        Eigen::Vector2<T> residualVector = predictedImagePoint - _observedImagePoint;

        std::copy(residualVector.data(), residualVector.data() + 2, residuals);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from the client code.
    static ceres::CostFunction *createCostFunction(const Point3D &fixedWorldPoint,
                                                   const Point2D &centeredImagePoint)
    {
        Eigen::Vector3<double> eigenWorldPoint(fixedWorldPoint.x(), fixedWorldPoint.y(), fixedWorldPoint.z());
        Eigen::Vector2<double> eigenImagePoint(centeredImagePoint.x(), centeredImagePoint.y());
        return (new ceres::AutoDiffCostFunction<FixedPointReprojectionError, 2, 8>(
            new FixedPointReprojectionError(eigenWorldPoint, eigenImagePoint)));
    }

    const Eigen::Vector3<double> _fixedWorldPoint;
    const Eigen::Vector2<double> _observedImagePoint;
};

struct CurvePointReprojectionError
{
    CurvePointReprojectionError(const std::vector<Eigen::Vector3<double>> &fixedWorldPolyline,
                                const Eigen::Vector2<double> &observedImagePoint)
        : _fixedWorldPolyline(fixedWorldPolyline),
          _observedImagePoint(observedImagePoint)
    {
        _cumulatedSegmentLengths = {0.0};
        for (int i = 1; i < fixedWorldPolyline.size(); i++)
        {
            _cumulatedSegmentLengths.emplace_back(_cumulatedSegmentLengths[i - 1] + (fixedWorldPolyline[i] - fixedWorldPolyline[i - 1]).norm());
        }
    }

    template <typename T>
    bool operator()(const T *const camera_p,
                    const T *const worldPointParameter_p,
                    T *residuals) const
    {
        const Eigen::Vector3<T> angleAxis(&camera_p[0]);
        const Eigen::Vector3<T> position(&camera_p[3]);
        const T focalLength = camera_p[6];
        const T k1 = camera_p[7];

        const Eigen::Vector3<T> worldPoint = getWorldPoint(worldPointParameter_p[0]) - position;

        Eigen::Vector3<T> cameraPoint;
        ceres::AngleAxisRotatePoint(angleAxis.data(), worldPoint.data(), cameraPoint.data());
        // cameraPoint += translation;
        Eigen::Vector2<T> normalizedPlaneProjection = cameraPoint.hnormalized();
        T r2 = normalizedPlaneProjection.squaredNorm(); // normalizedPlaneProjectionRadiusSquared
        T distortion = 1.0 + k1 * r2;
        Eigen::Vector2<T> predictedImagePoint = focalLength * distortion * normalizedPlaneProjection;
        Eigen::Vector2<T> residualVector = predictedImagePoint - _observedImagePoint;

        std::copy(residualVector.data(), residualVector.data() + 2, residuals);
        return true;
    }

    template <typename T>
    Eigen::Vector3<T> getWorldPoint(const T &curveParameter) const
    {
        size_t low = 0;
        size_t high = _cumulatedSegmentLengths.size() - 1;

        // Binary search to find the interval
        while (low < high - 1)
        {
            size_t mid = (low + high) / 2;
            if (_cumulatedSegmentLengths[mid] < curveParameter)
            {
                low = mid;
            }
            else
            {
                high = mid;
            }
        }

        const auto p1 = _fixedWorldPolyline[low].cast<T>();
        const auto p2 = _fixedWorldPolyline[high].cast<T>();
        const auto t = (curveParameter - _cumulatedSegmentLengths[low]) / (_cumulatedSegmentLengths[high] - _cumulatedSegmentLengths[low]);
        return (T(1.0) - t) * p1 + t * p2;
    }

    // Factory to hide the construction of the CostFunction object from the client code.
    static ceres::CostFunction *createCostFunction(const Polyline3D &fixedWorldPolyline,
                                                   const Point2D &centeredImagePoint)
    {
        std::vector<Eigen::Vector3<double>> eigenWorldPolyline;
        for (const auto &point : fixedWorldPolyline)
        {
            eigenWorldPolyline.emplace_back(Eigen::Vector3<double>(point.x(), point.y(), point.z()));
        }
        Eigen::Vector2<double> eigenImagePoint(centeredImagePoint.x(), centeredImagePoint.y());
        return (new ceres::AutoDiffCostFunction<CurvePointReprojectionError, 2, 8, 1>(
            new CurvePointReprojectionError(eigenWorldPolyline, eigenImagePoint)));
    }

    const std::vector<Eigen::Vector3<double>> _fixedWorldPolyline;
    const Eigen::Vector2<double> _observedImagePoint;
    std::vector<double> _cumulatedSegmentLengths;
};

struct CameraSoftConstraintResidual
{
    CameraSoftConstraintResidual(const Camera &previousCamera)
    {
        auto camPos = previousCamera.getPosition();

        _cameraX = camPos[0];
        _cameraY = camPos[1];
        _cameraZ = camPos[2];
    }

    template <typename T>
    bool operator()(const T *camera, T *residuals) const
    {

        residuals[0] = (camera[3] - T(_cameraX));
        residuals[1] = (camera[4] - T(_cameraY));
        residuals[2] = (camera[5] - T(_cameraZ));

        return true;
    }

    double _cameraX, _cameraY, _cameraZ;
};

struct CameraSoftOpticalAxisConstraintResidual
{
    static constexpr double MINIMUM_NORMALIZATION_RADIUS = 1.0e-12;

    CameraSoftOpticalAxisConstraintResidual(const Point3D &tripodRotationCenter, double offsetToSensor)
        : _tripod(tripodRotationCenter.hx(), tripodRotationCenter.hy(), tripodRotationCenter.hz())
    {
        _radius = offsetToSensor;
    }

    template <typename T>
    bool operator()(const T *camera, T *residuals) const
    {
        const Eigen::Vector3<T> angleAxis(&camera[0]);
        const Eigen::Vector3<T> cameraEye(&camera[3]);

        Eigen::Matrix3<T> rotation;
        // ceres expects R, not Rc, but expects column major (therefore Rc in row major is provided)

        ceres::AngleAxisToRotationMatrix(camera, rotation.data());

        // last row of rotation matrix is lookat
        Eigen::Vector3<T> lookAt(rotation(2, 0), rotation(2, 1), rotation(2, 2));

        Eigen::Vector3<T> eyeToTripodVector = _tripod.cast<T>() - cameraEye;

        Eigen::Vector3<T> t = cameraEye + lookAt.dot(eyeToTripodVector) / lookAt.dot(lookAt) * lookAt;

        const T squaredDistance = (t - _tripod.cast<T>()).squaredNorm();

        // Same zero set and local scale as radius - distance, without the
        // square root derivative at zero. The normalization is undefined for a
        // zero radius, where the squared distance already has the right zero.
        if (std::abs(_radius) <= MINIMUM_NORMALIZATION_RADIUS)
        {
            residuals[0] = squaredDistance;
        }
        else
        {
            residuals[0] = (squaredDistance - T(_radius * _radius)) / T(2.0 * _radius);
        }

        return true;
    }

    const Eigen::Vector3<double> _tripod;

    double _tripodX, _tripodY, _tripodZ, _radius;
};
