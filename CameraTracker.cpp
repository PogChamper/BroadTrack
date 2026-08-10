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

#include "CameraTracker.h"
#include "Residuals.h"
#include "LineIoUScore.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>


namespace
{
constexpr double NUMERICAL_EPSILON = 1.0e-12;
constexpr double ROOT_DUPLICATE_RELATIVE_TOLERANCE = 1.0e-9;

struct ReprojectionEvaluation
{
    size_t validCount = 0;
    size_t inlierCount = 0;
    double inlierMean = std::numeric_limits<double>::infinity();
    double truncatedMean = 0.0;
};

struct HypothesisEvaluation
{
    ReprojectionEvaluation reprojection;
    size_t firstPointId = std::numeric_limits<size_t>::max();
    size_t secondPointId = std::numeric_limits<size_t>::max();
    bool isBaseline = false;
};

bool isFinite(const Point2D &point)
{
    return std::isfinite(point.w()) &&
           std::abs(point.w()) > NUMERICAL_EPSILON &&
           std::isfinite(point.x()) &&
           std::isfinite(point.y());
}

bool isFinite(const Point3D &point)
{
    return std::isfinite(point.w()) &&
           std::abs(point.w()) > NUMERICAL_EPSILON &&
           std::isfinite(point.x()) &&
           std::isfinite(point.y()) &&
           std::isfinite(point.z());
}

bool findClosestFiniteObservation(const Point2D &projectedPoint,
                                  const std::vector<Point2D> &observations,
                                  Point2D &closestObservation,
                                  double &closestDistance)
{
    closestDistance = std::numeric_limits<double>::infinity();
    bool found = false;
    for (const auto &observation : observations)
    {
        if (!isFinite(observation))
        {
            continue;
        }

        const double distance = projectedPoint.distance(observation);
        if (std::isfinite(distance) && distance < closestDistance)
        {
            closestDistance = distance;
            closestObservation = observation;
            found = true;
        }
    }
    return found;
}

ReprojectionEvaluation evaluateReprojections(
    const SoccerPitch3D &soccerPitch,
    const std::vector<std::pair<SoccerPitch3D::PointID, std::vector<Point2D>>> &points,
    int threshold,
    std::vector<bool> &outInliers,
    const Camera &camera)
{
    outInliers.assign(points.size(), false);

    ReprojectionEvaluation evaluation;
    const double truncation = std::max(1.0, static_cast<double>(threshold));
    double truncatedErrorSum = 0.0;
    double inlierErrorSum = 0.0;

    for (size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex)
    {
        double error = truncation;
        Point2D projectedPoint;
        const Point3D worldPoint = soccerPitch.getPoint3D(points[pointIndex].first);
        const bool validProjection = isFinite(worldPoint) &&
                                     camera.project(worldPoint, projectedPoint, true) &&
                                     isFinite(projectedPoint);

        if (validProjection)
        {
            Point2D closestObservation;
            double closestDistance;
            if (findClosestFiniteObservation(projectedPoint,
                                             points[pointIndex].second,
                                             closestObservation,
                                             closestDistance))
            {
                ++evaluation.validCount;
                error = closestDistance;
                if (closestDistance <= static_cast<double>(threshold))
                {
                    outInliers[pointIndex] = true;
                    ++evaluation.inlierCount;
                    inlierErrorSum += closestDistance;
                }
            }
        }

        truncatedErrorSum += std::min(error, truncation);
    }

    evaluation.truncatedMean = points.empty()
                                   ? truncation
                                   : truncatedErrorSum / points.size();
    if (evaluation.inlierCount > 0)
    {
        evaluation.inlierMean = inlierErrorSum / evaluation.inlierCount;
    }
    return evaluation;
}

bool isBetterHypothesis(const HypothesisEvaluation &candidate,
                        const HypothesisEvaluation &reference)
{
    if (candidate.reprojection.inlierCount != reference.reprojection.inlierCount)
    {
        return candidate.reprojection.inlierCount > reference.reprojection.inlierCount;
    }
    if (candidate.reprojection.inlierMean != reference.reprojection.inlierMean)
    {
        return candidate.reprojection.inlierMean < reference.reprojection.inlierMean;
    }
    if (candidate.reprojection.truncatedMean != reference.reprojection.truncatedMean)
    {
        return candidate.reprojection.truncatedMean < reference.reprojection.truncatedMean;
    }

    // Prefer the current camera on an exact tie. Pair IDs make ties between
    // generated hypotheses independent of iteration order.
    if (candidate.isBaseline != reference.isBaseline)
    {
        return candidate.isBaseline;
    }
    if (candidate.firstPointId != reference.firstPointId)
    {
        return candidate.firstPointId < reference.firstPointId;
    }
    return candidate.secondPointId < reference.secondPointId;
}
} // namespace

// Map a pixel center of the source resolution onto the pixel center of the
// target resolution, with independent horizontal and vertical scales.
cv::Point2d mapPixelCenter(const cv::Point2d &point, const cv::Size &sourceResolution, const cv::Size &targetResolution)
{
    return cv::Point2d((point.x + 0.5) * targetResolution.width / sourceResolution.width - 0.5,
                       (point.y + 0.5) * targetResolution.height / sourceResolution.height - 0.5);
}

Point3D closestPointOnSegment(const Point3D &P, const Point3D &A, const Point3D &B)
{
    Point3D AB = B - A;
    Point3D AP = P - A;
    double t = AP.dotProduct(AB) / AB.squaredNorm();
    if (t < 0.0)
    {
        return A;
    }
    else if (t > 1.0)
    {
        return B;
    }
    else
    {
        return A + AB * t;
    }
}

double getCurveParameter(const Point3D &point3D, const Polyline3D &polyline3D)
{
    double minSquaredDistance = std::numeric_limits<double>::max();
    double closestCurvePointParameter;
    double t = 0.0;
    for (int i = 0; i < polyline3D.size() - 1; i++)
    {
        auto segmentStart = polyline3D[i];
        auto segmentEnd = polyline3D[i + 1];
        Point3D closestPoint = closestPointOnSegment(point3D, segmentStart, segmentEnd);
        double sqDist = point3D.squaredDistance(closestPoint);
        if (sqDist < minSquaredDistance)
        {
            minSquaredDistance = sqDist;
            closestCurvePointParameter = t + segmentStart.distance(closestPoint);
        }
        t += segmentStart.distance(segmentEnd);
    }
    return closestCurvePointParameter;
}

CameraTracker::CameraTracker() : _pointExtractor(0.5, 10)
{
    _lambda = 0.;
    _soccerPitch3D = SoccerPitch3D();
}

std::tuple<double, Camera> CameraTracker::update(const cv::Mat &semLinesMask,
                                                 const std::vector<std::pair<Point3D, Point2D>> &pitchProjections,
                                                 bool softPosition,
                                                 bool ptz,
                                                 bool lensDistortion,
                                                 int cauchyParameter)
{
    Point2D principalPoint = _camera.getPrincipalPoint();

    _pointExtractor.setMask(semLinesMask);
    std::map<int, std::vector<cv::Point2d>> soccerLineMatches;
    _pointExtractor.getExtractedSubpixelPoints(soccerLineMatches);
    const cv::Size cameraResolution = _camera.getPixelResolution();
    for (auto &labeledLine : soccerLineMatches)
    {
        for (auto &point : labeledLine.second)
        {
            point = mapPixelCenter(point, semLinesMask.size(), cameraResolution);
        }
    }

    ceres::Problem problem;

    std::array<double, 8> cameraData;
    auto angleAxisVector = _camera.getAngleAxis();
    cameraData[0] = angleAxisVector[0];
    cameraData[1] = angleAxisVector[1];
    cameraData[2] = angleAxisVector[2];
    auto positionVector = _camera.getPosition();
    cameraData[3] = positionVector[0];
    cameraData[4] = positionVector[1];
    cameraData[5] = positionVector[2];
    auto focalLength = _camera.getFocalLength();
    cameraData[6] = focalLength;
    auto radialDistortion = _camera.getRadialDistortion();
    if (radialDistortion.size())
    {
        cameraData[7] = radialDistortion[0];
    }
    else
    {
        cameraData[7] = 0.;
    }
    // Fixed radial
    if (!lensDistortion)
    {
        cameraData[7] = 0.;
    }

    problem.AddParameterBlock(&cameraData[0], 8);
    // No radial
    if (!lensDistortion)
    {
        auto *fixedk1 = new ceres::SubsetParameterization(8, {7});
        problem.SetParameterization(&cameraData[0], fixedk1);
    }

    std::map<SoccerPitch3D::LineID, std::vector<double>> curvePointParameterData;

    for (const auto &labeledLine : soccerLineMatches)
    {
        if (labeledLine.first == SoccerPitch3D::LineID::UNDEFINED_LINE)
        {
            continue;
        }
        auto lineId = static_cast<SoccerPitch3D::LineID>(labeledLine.first);
        auto curveSample = labeledLine.second;
        auto polyLine3D = _soccerPitch3D.getPolyline3D(lineId, 1.0); // 0.01, 0.1
        for (const auto &curvePoint : curveSample)
        {
            auto point3D = _soccerPitch3D.getSurface().intersection(
                _camera.getRay(Point2D(curvePoint.x, curvePoint.y)));
            auto t = getCurveParameter(point3D, polyLine3D);
            curvePointParameterData[lineId].emplace_back(t);
        }
    }
    for (const auto &labeledLine : soccerLineMatches)
    {
        auto lineId = static_cast<SoccerPitch3D::LineID>(labeledLine.first);
        auto curveSample = labeledLine.second;
        auto polyLine3D = _soccerPitch3D.getPolyline3D(lineId, 1.0);

        for (int i = 0; i < curveSample.size(); i++)
        {
            auto curvePoint = Point2D(curveSample[i].x, curveSample[i].y);
            ceres::CostFunction *reprojectionCostFunction =
                CurvePointReprojectionError::createCostFunction(polyLine3D, curvePoint - principalPoint);
            problem.AddResidualBlock(reprojectionCostFunction,
                                     new ceres::CauchyLoss(cauchyParameter),
                                     &cameraData[0],
                                     &curvePointParameterData.at(lineId)[i]);
        }
    }

    for (const auto &pp : pitchProjections)
    {
        ceres::CostFunction *reprojectionCostFunction =
            FixedPointReprojectionError::createCostFunction(pp.first, pp.second - principalPoint);
        problem.AddResidualBlock(reprojectionCostFunction,
                                 new ceres::CauchyLoss(1.0),
                                 &cameraData[0]);
    }

    if (softPosition)
    {
        auto weightedLossTripod = new ceres::ScaledLoss(nullptr, 150, ceres::TAKE_OWNERSHIP);
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CameraSoftOpticalAxisConstraintResidual, 1, 8>(
                new CameraSoftOpticalAxisConstraintResidual(_tripodCenter, _tripodRadius)),
            weightedLossTripod,
            &cameraData[0]);
    }

    ceres::Solver::Options options;
    options.num_threads = 32;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    angleAxisVector = {cameraData[0], cameraData[1], cameraData[2]};
    Matrix3x3 orientationMatrix;
    ceres::AngleAxisToRotationMatrix(angleAxisVector.data(), orientationMatrix.data());
    positionVector = {cameraData[3], cameraData[4], cameraData[5]};
    focalLength = cameraData[6];

    radialDistortion = {cameraData[7]};

    _camera.setOrientation(orientationMatrix)
        .setPosition(positionVector)
        .setPrincipalPoint(principalPoint)
        .setFocalLength(focalLength)
        .setRadialDistortion(radialDistortion);

    double outScore = this->evaluate(semLinesMask);

    return std::tuple<double, Camera>(outScore, _camera);
}

double CameraTracker::evaluate(const cv::Mat &semLinesMask)
{

    cv::Mat rawLines = semLinesMask.clone();
    rawLines.setTo(255, rawLines >= 150);
    rawLines.setTo(0, rawLines < 150);
    cv::resize(rawLines, rawLines, cv::Size(960, 540), 1);
    // Scale the projection from camera pixels to the 960x540 score mask.
    const cv::Size resolution = _camera.getPixelResolution();
    Matrix3x3 H = _camera.getGroundPlaneHomography();
    H = Matrix3x3(960.0 / resolution.width, 0, 0, 0, 540.0 / resolution.height, 0, 0, 0, 1).multiply(H);
    double outScore = LineIoUScore(rawLines).evaluateFast(
        H,
        _soccerPitch3D.getLength(),
        _soccerPitch3D.getWidth());
    return outScore;
}

Camera CameraTracker::getCamera() const
{
    return _camera;
}

void CameraTracker::setCamera(const Camera &camera)
{
    _camera = camera;
}

void CameraTracker::setTripodInfo(const Point3D &center, double radius)
{
    _tripodCenter = center;
    _tripodRadius = radius;
}

std::tuple<double, double> CameraTracker::evaluateReprojectionError(const std::vector<std::pair<SoccerPitch3D::PointID,
                                                                                                std::vector<Point2D>>> &points,
                                                                    int threshold,
                                                                    std::vector<bool> &outInliers,
                                                                    const Camera &camera)
{
    const ReprojectionEvaluation evaluation = evaluateReprojections(
        _soccerPitch3D, points, threshold, outInliers, camera);
    return std::make_tuple(evaluation.truncatedMean, evaluation.inlierMean);
}

void appendDistinctPositiveRoot(std::vector<double> &roots, double root)
{
    if (!std::isfinite(root) || root <= NUMERICAL_EPSILON)
    {
        return;
    }

    for (const double existingRoot : roots)
    {
        const double scale = std::max(1.0, std::max(std::abs(root), std::abs(existingRoot)));
        if (std::abs(root - existingRoot) <= ROOT_DUPLICATE_RELATIVE_TOLERANCE * scale)
        {
            return;
        }
    }
    roots.push_back(root);
}

std::vector<double> squaredFocalLengthFromTwoPoints(double a, double b, double c, double d)
{
    std::vector<double> roots;
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) || !std::isfinite(d))
    {
        return roots;
    }

    const double dSquared = d * d;
    const double A = dSquared - 1.0;
    const double B = dSquared * (a + b) - 2.0 * c;
    const double C = dSquared * a * b - c * c;
    if (!std::isfinite(A) || !std::isfinite(B) || !std::isfinite(C))
    {
        return roots;
    }

    const double coefficientScale = std::max(1.0, std::max(std::abs(A), std::max(std::abs(B), std::abs(C))));
    if (std::abs(A) <= NUMERICAL_EPSILON * coefficientScale)
    {
        // |d| ~= 1 means the two world rays do not constrain focal length.
        return roots;
    }

    double discriminant = B * B - 4.0 * A * C;
    const double discriminantScale = std::max(1.0, std::max(B * B, std::abs(4.0 * A * C)));
    if (!std::isfinite(discriminant) || discriminant < -NUMERICAL_EPSILON * discriminantScale)
    {
        return roots;
    }
    discriminant = std::max(0.0, discriminant);

    const double squareRootDiscriminant = std::sqrt(discriminant);
    const double q = -0.5 * (B + std::copysign(squareRootDiscriminant, B));
    if (std::abs(q) > NUMERICAL_EPSILON * coefficientScale)
    {
        appendDistinctPositiveRoot(roots, q / A);
        appendDistinctPositiveRoot(roots, C / q);
    }
    else if (std::abs(B) > NUMERICAL_EPSILON * coefficientScale)
    {
        // Numerically repeated root; q would divide by zero.
        appendDistinctPositiveRoot(roots, -B / (2.0 * A));
    }

    std::sort(roots.begin(), roots.end());
    return roots;
}

std::vector<double> estimateFocalLengthsFromPositionAndTwoPoints(const std::vector<std::pair<Point3D, Point2D>> &points, const Point3D &position,
                                                                 const Point2D &principalPoint, const cv::Size &resolution)
{
    std::vector<double> focalLengths;
    if (points.size() != 2 || !isFinite(position) || !isFinite(principalPoint) || resolution.width <= 0 || resolution.height <= 0 ||
        !isFinite(points[0].first) || !isFinite(points[1].first) || !isFinite(points[0].second) || !isFinite(points[1].second))
    {
        return focalLengths;
    }

    Point3D X1 = points[0].first - position;
    Point3D X2 = points[1].first - position;
    const double firstRayNorm = X1.norm();
    const double secondRayNorm = X2.norm();
    if (!std::isfinite(firstRayNorm) || !std::isfinite(secondRayNorm) || firstRayNorm <= NUMERICAL_EPSILON || secondRayNorm <= NUMERICAL_EPSILON)
    {
        return focalLengths;
    }
    X1.scale(1.0 / firstRayNorm);
    X2.scale(1.0 / secondRayNorm);
    double d = X1.dotProduct(X2) - 1.0;
    if (!std::isfinite(d) || std::abs(d) > 1.0 + NUMERICAL_EPSILON || 1.0 - std::abs(d) <= NUMERICAL_EPSILON)
    {
        return focalLengths;
    }
    d = std::max(-1.0, std::min(1.0, d));

    // The closed-form solver works in principal-centred image coordinates
    // normalized by the current image height.
    const double normalizationScale = static_cast<double>(resolution.height);
    const Point2D x1 = (points[0].second - principalPoint) / normalizationScale;
    const Point2D x2 = (points[1].second - principalPoint) / normalizationScale;
    if (x1.distance(x2) <= NUMERICAL_EPSILON)
    {
        return focalLengths;
    }

    const double a = x1.dotProduct(x1) - 1.0;
    const double b = x2.dotProduct(x2) - 1.0;
    const double c = x1.dotProduct(x2) - 1.0;

    // Keep the roots within a broad 5..150 degree horizontal field of view.
    constexpr double PI = 3.14159265358979323846;
    const double minimumHorizontalFieldOfView = 5.0 * PI / 180.0;
    const double maximumHorizontalFieldOfView = 150.0 * PI / 180.0;
    const double minimumFocalLength = resolution.width / (2.0 * std::tan(maximumHorizontalFieldOfView / 2.0));
    const double maximumFocalLength = resolution.width / (2.0 * std::tan(minimumHorizontalFieldOfView / 2.0));

    const std::vector<double> squaredNormalizedFocalLengths = squaredFocalLengthFromTwoPoints(a, b, c, d);
    for (const double squaredNormalizedFocalLength : squaredNormalizedFocalLengths)
    {
        const double focalLength = std::sqrt(squaredNormalizedFocalLength) * normalizationScale;
        if (std::isfinite(focalLength) && focalLength >= minimumFocalLength && focalLength <= maximumFocalLength)
        {
            focalLengths.push_back(focalLength);
        }
    }
    return focalLengths;
}

std::tuple<double, double> CameraTracker::estimatePanTilt(const std::vector<std::pair<Point3D, Point2D>> &detectedPoints,
                                                          double focal,
                                                          Point3D position,
                                                          size_t iwidth,
                                                          size_t iheight)
{
    const double invalidAngle = std::numeric_limits<double>::quiet_NaN();
    if (detectedPoints.empty() || !std::isfinite(focal) || focal <= 0.0 || !isFinite(position))
    {
        return std::make_tuple(invalidAngle, invalidAngle);
    }

    const cv::Size currentResolution = _camera.getPixelResolution();
    if (currentResolution.width != static_cast<int>(iwidth) || currentResolution.height != static_cast<int>(iheight))
    {
        return std::make_tuple(invalidAngle, invalidAngle);
    }

    Point3D mean_optical_axis(0., 0., 0.);
    for (auto &point : detectedPoints)
    {
        if (!isFinite(point.first) || !isFinite(point.second))
        {
            return std::make_tuple(invalidAngle, invalidAngle);
        }
        Point3D P = point.first - position;
        mean_optical_axis = mean_optical_axis + P;
    }

    const double opticalAxisNorm = mean_optical_axis.norm();
    if (!std::isfinite(opticalAxisNorm) || opticalAxisNorm <= NUMERICAL_EPSILON)
    {
        return std::make_tuple(invalidAngle, invalidAngle);
    }
    mean_optical_axis.scale(1.0 / opticalAxisNorm);
    double curr_pan = atan2(mean_optical_axis[0], -mean_optical_axis[1]);
    double curr_tilt = atan2(-mean_optical_axis[1], mean_optical_axis[2]);
    if (!std::isfinite(curr_pan) || !std::isfinite(curr_tilt))
    {
        return std::make_tuple(invalidAngle, invalidAngle);
    }

    // Copy the current camera so that distortion, resolution and every other
    // non-pose attribute survive the estimation.
    Camera camera = _camera;

    camera.setPanTiltRoll(Vector3x1(curr_pan, curr_tilt, 0.))
        .setPosition(Vector3x1(position.x(), position.y(), position.z()))
        .setFocalLength(focal);

    for (int i = 0; i < 5; i++)
    {
        std::vector<double> pans;
        std::vector<double> tilts;
        for (auto &point : detectedPoints)
        {
            Point2D projected;
            camera.project(point.first, projected, false);
            if (!isFinite(projected))
            {
                return std::make_tuple(invalidAngle, invalidAngle);
            }
            double dx = point.second.x() - projected.x();
            double dy = projected.y() - point.second.y();
            double dpan = atan2(dx, focal);
            double dtilt = atan2(dy, focal);
            if (!std::isfinite(dpan) || !std::isfinite(dtilt))
            {
                return std::make_tuple(invalidAngle, invalidAngle);
            }
            pans.push_back(dpan);
            tilts.push_back(dtilt);
        }
        double meanPan = std::accumulate(pans.begin(), pans.end(), 0.0) / pans.size();
        curr_pan -= meanPan;
        double meanTilt = std::accumulate(tilts.begin(), tilts.end(), 0.0) / tilts.size();
        curr_tilt -= meanTilt;
        if (!std::isfinite(curr_pan) || !std::isfinite(curr_tilt))
        {
            return std::make_tuple(invalidAngle, invalidAngle);
        }

        camera.setPanTiltRoll(Vector3x1(curr_pan, curr_tilt, 0.))
            .setPosition(Vector3x1(position.x(), position.y(), position.z()))
            .setFocalLength(focal);
    }

    return std::tie(curr_pan, curr_tilt);
}

void CameraTracker::reinit(const std::vector<std::pair<SoccerPitch3D::PointID, std::vector<Point2D>>> &detectedPoints, int threshold, int n_iterations)
{
    if (detectedPoints.size() < 2 || n_iterations == 0)
    {
        return;
    }

    std::vector<bool> outInliers;
    HypothesisEvaluation bestEvaluation;
    bestEvaluation.reprojection = evaluateReprojections(_soccerPitch3D, detectedPoints, threshold, outInliers, _camera);
    bestEvaluation.isBaseline = true;
    Camera bestCamera = _camera;
    const cv::Size resolution = _camera.getPixelResolution();
    const Point2D principalPoint = _camera.getPrincipalPoint();

    const size_t totalPairs = detectedPoints.size() * (detectedPoints.size() - 1) / 2;
    const size_t pairBudget = n_iterations > 0 ? std::min(static_cast<size_t>(n_iterations), totalPairs) : totalPairs;
    size_t examinedPairs = 0;
    // Enumerate every unordered pair once, by increasing index gap so a
    // positive cap still spreads the budget across all detections.
    for (size_t pointGap = 1; pointGap < detectedPoints.size() && examinedPairs < pairBudget; ++pointGap)
    {
        for (size_t id1 = 0; id1 + pointGap < detectedPoints.size() && examinedPairs < pairBudget; ++id1)
        {
            const size_t id2 = id1 + pointGap;
            ++examinedPairs;
            if (detectedPoints[id1].second.empty() || detectedPoints[id2].second.empty())
            {
                continue;
            }

            std::vector<std::pair<Point3D, Point2D>> candidates;
            candidates.push_back(std::make_pair(_soccerPitch3D.getPoint3D(detectedPoints[id1].first), detectedPoints[id1].second[0]));
            candidates.push_back(std::make_pair(_soccerPitch3D.getPoint3D(detectedPoints[id2].first), detectedPoints[id2].second[0]));

            const std::vector<double> focalLengths = estimateFocalLengthsFromPositionAndTwoPoints(candidates, _tripodCenter, principalPoint, resolution);
            for (const double focalLength : focalLengths)
            {
                double pan;
                double tilt;
                std::tie(pan, tilt) = this->estimatePanTilt(candidates, focalLength, _tripodCenter, resolution.width, resolution.height);
                if (!std::isfinite(pan) || !std::isfinite(tilt))
                {
                    continue;
                }

                // Start from the current camera so resolution and distortion carry over.
                Camera hypothesis = _camera;
                hypothesis.setPanTiltRoll(Vector3x1(pan, tilt, 0.))
                    .setPosition(Vector3x1(_tripodCenter.x(), _tripodCenter.y(), _tripodCenter.z()))
                    .setFocalLength(focalLength);

                HypothesisEvaluation hypothesisEvaluation;
                hypothesisEvaluation.reprojection = evaluateReprojections(_soccerPitch3D, detectedPoints, threshold, outInliers, hypothesis);
                hypothesisEvaluation.firstPointId = id1;
                hypothesisEvaluation.secondPointId = id2;
                if (hypothesisEvaluation.reprojection.validCount < 2)
                {
                    continue;
                }

                Camera selectedCamera = hypothesis;
                HypothesisEvaluation selectedEvaluation = hypothesisEvaluation;

                if (hypothesisEvaluation.reprojection.inlierCount > 0.5 * detectedPoints.size() && hypothesisEvaluation.reprojection.inlierCount > 3)
                {
                    std::vector<std::pair<Point3D, Point2D>> inliers;
                    for (size_t pointIndex = 0; pointIndex < outInliers.size(); ++pointIndex)
                    {
                        if (!outInliers[pointIndex])
                        {
                            continue;
                        }

                        Point2D projectedPoint;
                        Point2D closestObservation;
                        double closestDistance;
                        const Point3D worldPoint = _soccerPitch3D.getPoint3D(detectedPoints[pointIndex].first);
                        if (hypothesis.project(worldPoint, projectedPoint, true) && isFinite(projectedPoint) &&
                            findClosestFiniteObservation(projectedPoint, detectedPoints[pointIndex].second, closestObservation, closestDistance))
                        {
                            inliers.push_back(std::make_pair(worldPoint, closestObservation));
                        }
                    }

                    if (inliers.size() >= 2)
                    {
                        double guidedPan;
                        double guidedTilt;
                        std::tie(guidedPan, guidedTilt) = this->estimatePanTilt(inliers, focalLength, _tripodCenter, resolution.width, resolution.height);
                        if (std::isfinite(guidedPan) && std::isfinite(guidedTilt))
                        {
                            Camera guidedHypothesis = _camera;
                            guidedHypothesis.setPanTiltRoll(Vector3x1(guidedPan, guidedTilt, 0.))
                                .setPosition(Vector3x1(_tripodCenter.x(), _tripodCenter.y(), _tripodCenter.z()))
                                .setFocalLength(focalLength);

                            HypothesisEvaluation guidedEvaluation;
                            guidedEvaluation.reprojection = evaluateReprojections(_soccerPitch3D, detectedPoints, threshold, outInliers, guidedHypothesis);
                            guidedEvaluation.firstPointId = id1;
                            guidedEvaluation.secondPointId = id2;

                            if (guidedEvaluation.reprojection.validCount >= 2 && isBetterHypothesis(guidedEvaluation, hypothesisEvaluation))
                            {
                                selectedCamera = guidedHypothesis;
                                selectedEvaluation = guidedEvaluation;
                            }
                        }
                    }
                }

                if (isBetterHypothesis(selectedEvaluation, bestEvaluation))
                {
                    bestEvaluation = selectedEvaluation;
                    bestCamera = selectedCamera;
                }
            }
        }
    }
    _camera = bestCamera;
}
