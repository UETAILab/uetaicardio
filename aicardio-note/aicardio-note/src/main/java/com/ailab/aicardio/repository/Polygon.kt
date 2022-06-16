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

package com.ailab.aicardio.repository

import android.graphics.Path
import android.graphics.Point
import android.graphics.RectF

/*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * Polygon
 * This class provides support for polygon shapes. It is based directly
 * on the java.awt.Polygon class available in Java.
 *
 * @see http://java.sun.com/j2se/1.4.2/docs/api/java/awt/Polygon.html
 *
 * @see http://developer.android.com/reference/android/graphics/Path.html
 *
 *
 * @author Paul Gregoire (mondain@gmail.com)
 */

class Polygon : Path {
    //private final static String tag = "Polygon";
    /**
     * Gets the bounding box of this Polygon
     */
    /**
     * Bounds of the polygon
     */
    var bounds: RectF? = null

    /**
     * Total number of points
     */
    var npoints = 0

    /**
     * Array of x coordinate
     */
    var xpoints: IntArray

    /**
     * Array of y coordinates
     */
    var ypoints: IntArray
    /**
     * Creates an empty polygon expecting a given number of points
     *
     * @param  numPoints the total number of points in the Polygon
     */
    /**
     * Creates an empty polygon with 16 points
     */
    @JvmOverloads
    constructor(numPoints: Int = 16) : super() {
        xpoints = IntArray(numPoints)
        ypoints = IntArray(numPoints)
    }

    /**
     * Constructs and initializes a Polygon from the specified parameters
     *
     * @param xpoints an array of x coordinates
     * @param ypoints an array of y coordinates
     * @param npoints the total number of points in the Polygon
     */
    constructor(xpoints: IntArray, ypoints: IntArray, npoints: Int) : super() {
        this.xpoints = xpoints
        this.ypoints = ypoints
        this.npoints = npoints
        moveTo(xpoints[0].toFloat(), ypoints[0].toFloat())
        for (p in 1 until npoints) {
            lineTo(xpoints[p].toFloat(), ypoints[p].toFloat())
        }
        close()
    }

    /**
     * Appends the specified coordinates to this Polygon. Remember to close
     * the polygon after adding all the points.
     */
    fun addPoint(x: Int, y: Int) {
        //Log.d(tag, "addPoint - x: " + x + " y: " + y + " num: " + npoints);
        xpoints[npoints] = x
        ypoints[npoints] = y
        if (npoints > 0) {
            lineTo(x.toFloat(), y.toFloat())
        } else {
            moveTo(x.toFloat(), y.toFloat())
        }
        npoints++
    }

    /**
     * Determines whether the specified Point is inside this Polygon
     *
     * @param p the specified Point to be tested
     * @return true if the Polygon contains the Point; false otherwise
     */
    operator fun contains(p: Point): Boolean {
        return if (bounds != null) {
            bounds!!.contains(p.x.toFloat(), p.y.toFloat())
        } else {
            false
        }
    }

    /**
     * Determines whether the specified coordinates are inside this Polygon
     *
     * @param x the specified x coordinate to be tested
     * @param y the specified y coordinate to be tested
     * @return true if this Polygon contains the specified coordinates, (x, y);
     * false otherwise
     */
    fun contains(x: Int, y: Int): Boolean {
        return if (bounds != null) {
            bounds!!.contains(x.toFloat(), y.toFloat())
        } else {
            false
        }
    }

    /**
     * Tests if the interior of this Polygon entirely contains the specified
     * set of rectangular coordinates
     *
     * @param x the x coordinate of the top-left corner of the specified set of
     * rectangular coordinates
     * @param y the y coordinate of the top-left corner of the specified set of
     * rectangular coordinates
     * @param w the width of the set of rectangular coordinates
     * @param h the height of the set of rectangular coordinates
     * @return
     */
    fun contains(x: Double, y: Double, w: Double, h: Double): Boolean {
        return if (bounds != null) {
            val fx = x.toFloat()
            val fy = y.toFloat()
            val fw = w.toFloat()
            val fh = h.toFloat()
            //not sure if math is correct here
            val that = Path()
            //start
            that.moveTo(fx, fy)
            //go right
            that.lineTo(fx + fw, fy)
            //go down
            that.lineTo(fx + fw, fy - fh)
            //go left
            that.lineTo(fx, fy - fh)
            //close
            that.close()
            //bounds holder
            val thatBounds = RectF()
            that.computeBounds(thatBounds, false)
            bounds!!.contains(thatBounds)
        } else {
            false
        }
    }

    /**
     * Tests if the interior of this Polygon entirely contains the specified
     * Rectangle
     *
     * @param r the specified RectF
     * @return true if this Polygon entirely contains the specified RectF;
     * false otherwise.
     */
    operator fun contains(r: RectF?): Boolean {
        return if (bounds != null) {
            bounds!!.contains(r)
        } else {
            false
        }
    }

    /**
     * Tests if the interior of this Polygon intersects the interior of a
     * specified set of rectangular coordinates
     *
     * @param x the x coordinate of the specified rectangular shape's top-left
     * corner
     * @param y the y coordinate of the specified rectangular shape's top-left
     * corner
     * @param w the width of the specified rectangular shape
     * @param h the height of the specified rectangular shape
     * @return
     */
    fun intersects(x: Double, y: Double, w: Double, h: Double): Boolean {
        return if (bounds != null) {
            val fx = x.toFloat()
            val fy = y.toFloat()
            val fw = w.toFloat()
            val fh = h.toFloat()
            //not sure if math is correct here
            val that = Path()
            //start
            that.moveTo(fx, fy)
            //go right
            that.lineTo(fx + fw, fy)
            //go down
            that.lineTo(fx + fw, fy - fh)
            //go left
            that.lineTo(fx, fy - fh)
            //close
            that.close()
            //bounds holder
            val thatBounds = RectF()
            RectF.intersects(bounds, thatBounds)
        } else {
            false
        }
    }

    /**
     * Tests if the interior of this Polygon intersects the interior of a
     * specified Rectangle
     *
     * @param r a specified RectF
     * @return true if this Polygon and the interior of the specified RectF
     * intersect each other; false otherwise
     */
    fun intersects(r: RectF?): Boolean {
        return if (bounds != null) {
            RectF.intersects(bounds, r)
        } else {
            false
        }
    }

    /**
     * Invalidates or flushes any internally-cached data that depends on the
     * vertex coordinates of this Polygon
     */
    fun invalidate() {
        reset()
        xpoints = IntArray(npoints)
        ypoints = IntArray(npoints)
        bounds = null
    }

    /**
     * Close the current contour and generate the bounds.
     */
    override fun close() {
        super.close()
        //create bounds for this polygon
        bounds = RectF()
        computeBounds(bounds, false)
    }


}

//fun main() {
//    print("Test")
//
//    val polygon = Polygon(4)
//    polygon.addPoint(0, 0)
//    polygon.addPoint(10, 0)
//    polygon.addPoint(10, 10)
//    polygon.addPoint(0, 10)
//
//    print(polygon.contains(Point(20, 20)))
//    print(polygon.contains(Point(5, 5)))
//    print(polygon.contains(Point(8, 1)))
//    print(polygon.contains(Point(-1, 10)))
//
//
//}