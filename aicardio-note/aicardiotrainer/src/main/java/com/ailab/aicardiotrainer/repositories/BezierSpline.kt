/*
 * Copyright 2020 UET-AILAB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.ailab.aicardiotrainer.repositories

import android.graphics.Path
import android.graphics.PathMeasure
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.util.logging.Logger

class BezierSpline(knots: Int) {
    private val mKnots: Int
    private val mX: FloatArray
    private val mY: FloatArray
    private val mPX1: FloatArray
    private val mPY1: FloatArray
    private val mPX2: FloatArray
    private val mPY2: FloatArray
    private var mResolved = false
    private var mResolver: ControlPointsResolver? = null

    /**
     * Gets knots count.
     */
    fun knots(): Int {
        return mKnots
    }

    /**
     * Gets segments count.
     */
    fun segments(): Int {
        return mKnots - 1
    }

    /**
     * Sets coordinates of knot.
     */
    operator fun set(knot: Int, x: Float, y: Float) {
        mX[knot] = x
        mY[knot] = y
        mResolved = false
    }

    /**
     * Sets x coordinate of knot.
     */
    fun x(knot: Int, x: Float) {
        mX[knot] = x
        mResolved = false
    }

    /**
     * Sets y coordinate of knot.
     */
    fun y(knot: Int, y: Float) {
        mY[knot] = y
        mResolved = false
    }

    /**
     * Gets x coordinate of knot.
     */
    fun x(knot: Int): Float {
        return mX[knot]
    }

    /**
     * Gets y coordinate of knot.
     */
    fun y(knot: Int): Float {
        return mY[knot]
    }

    /**
     * Gets resolved x coordinate of first control point.
     */
    fun px1(segment: Int): Float {
        ensureResolved()
        return mPX1[segment]
    }

    /**
     * Gets resolved y coordinate of first control point.
     */
    fun py1(segment: Int): Float {
        ensureResolved()
        return mPY1[segment]
    }

    /**
     * Gets resolved x coordinate of second control point.
     */
    fun px2(segment: Int): Float {
        ensureResolved()
        return mPX2[segment]
    }

    /**
     * Gets resolved y coordinate of second control point.
     */
    fun py2(segment: Int): Float {
        ensureResolved()
        return mPY2[segment]
    }

    /**
     * Applies resolved control points to the specified Path.
     */
    fun applyToPath(path: Path) {
        ensureResolved()
        path.reset()
        path.moveTo(mX[0], mY[0])
        val segments = mKnots - 1
        if (segments == 1) {
            path.lineTo(mX[1], mY[1])
        } else {
            for (segment in 0 until segments) {
                val knot = segment + 1
                path.cubicTo(
                    mPX1[segment],
                    mPY1[segment],
                    mPX2[segment],
                    mPY2[segment],
                    mX[knot],
                    mY[knot]
                )
            }
        }
    }

    private fun applyToKnots(knots: JSONArray): Path {
        val path = Path()
        for (i in 0 until knots.length()) {
            try {
                set(
                    i,
                    knots.getJSONObject(i).getDouble("x").toFloat(),
                    knots.getJSONObject(i).getDouble("y").toFloat()
                )
            } catch (e: JSONException) {
                e.printStackTrace()
            }
        }
        applyToPath(path)
        return path
    }

    private fun pathToBoundary(path: Path, length: Int): JSONArray {
        val pathMeasure = PathMeasure(path, false)
        val boundary = JSONArray()
        for (i in 0..length) {
            val aCoordinates = floatArrayOf(0f, 0f)
            val distance = pathMeasure.length * i / length
            pathMeasure.getPosTan(distance, aCoordinates, null)
            val point = JSONObject()
            try {
                point.put("x", aCoordinates[0])
                point.put("y", aCoordinates[1])
                Logger.getLogger("DATA").warning(point.toString())
                boundary.put(point)
            } catch (e: JSONException) {
                e.printStackTrace()
            }
        }
        return boundary
    }

    fun apply(knots: JSONArray, length: Int): JSONArray {
        val path = applyToKnots(knots)
        return pathToBoundary(path, length)
    }

    private fun ensureResolved() {
        if (!mResolved) {
            val segments = mKnots - 1
            if (segments == 1) {
                mPX1[0] = mX[0]
                mPY1[0] = mY[0]
                mPX2[0] = mX[1]
                mPY2[0] = mY[1]
            } else {
                if (mResolver == null) {
                    mResolver = ControlPointsResolver(segments)
                }
                mResolver!!.resolve(mX, mPX1, mPX2)
                mResolver!!.resolve(mY, mPY1, mPY2)
            }
            mResolved = true
        }
    }

    /**
     * Copied from https://www.particleincell.com/wp-content/uploads/2012/06/bezier-spline.js
     */
    private class ControlPointsResolver internal constructor(private val mSegments: Int) {
        private val mA: FloatArray
        private val mB: FloatArray
        private val mC: FloatArray
        private val mR: FloatArray
        fun resolve(
            K: FloatArray,
            P1: FloatArray,
            P2: FloatArray
        ) {
            val segments = mSegments
            val last = segments - 1
            val A = mA
            val B = mB
            val C = mC
            val R = mR

            // prepare left most segment.
            A[0] = 0f
            B[0] = 2f
            C[0] = 1f
            R[0] = K[0] + 2f * K[1]

            // prepare internal segments.
            for (i in 1 until last) {
                A[i] = 1f
                B[i] = 4f
                C[i] = 1f
                R[i] = 4f * K[i] + 2f * K[i + 1]
            }

            // prepare right most segment.
            A[last] = 2f
            B[last] = 7f
            C[last] = 0f
            R[last] = 8f * K[last] + K[segments]

            // solves Ax=b with the Thomas algorithm (from Wikipedia).
            for (i in 1 until segments) {
                val m = A[i] / B[i - 1]
                B[i] = B[i] - m * C[i - 1]
                R[i] = R[i] - m * R[i - 1]
            }
            P1[last] = R[last] / B[last]
            for (i in segments - 2 downTo 0) {
                P1[i] = (R[i] - C[i] * P1[i + 1]) / B[i]
            }

            // we have p1, now compute p2.
            for (i in 0 until last) {
                P2[i] = 2f * K[i + 1] - P1[i + 1]
            }
            P2[last] = (K[segments] + P1[segments - 1]) / 2f
        }

        init {
            mA = FloatArray(mSegments)
            mB = FloatArray(mSegments)
            mC = FloatArray(mSegments)
            mR = FloatArray(mSegments)
        }
    }

    init {
        require(knots > 1) { "At least two knot points required" }
        mKnots = knots
        mX = FloatArray(knots)
        mY = FloatArray(knots)
        val segments = knots - 1
        mPX1 = FloatArray(segments)
        mPY1 = FloatArray(segments)
        mPX2 = FloatArray(segments)
        mPY2 = FloatArray(segments)
    }
}