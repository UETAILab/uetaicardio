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

package com.uetailab.aipacs.home_pacs.fragment_intepretation


import android.app.Activity
import android.app.Dialog
import android.content.Context
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet
import com.github.mikephil.charting.utils.ColorTemplate
import com.uetailab.aipacs.R
import kotlinx.android.synthetic.main.interpretation_view_dialog_gls_bar_chart.*
import kotlin.random.Random


class InterpretationViewGLSBarChartDialog(val activity: Activity, val interpretationViewVM: InterpretationViewVM) : Dialog(activity) {
    // example: https://github.com/PhilJay/MPAndroidChart/blob/master/MPChartExample/src/main/java/com/xxmassdeveloper/mpchartexample/MultiLineChartActivity.java

    companion object {
        const val TAG = "InterpretationViewGLSBarChartDialog"
    }
    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)

        val displayRectangle = Rect()
        val window: Window = activity.window
        window.decorView.getWindowVisibleDisplayFrame(displayRectangle)

        // inflate and adjust layout
        val inflater : LayoutInflater = activity.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        val layout: View = inflater.inflate(R.layout.interpretation_view_dialog_gls_bar_chart, null)
//
        layout.minimumWidth = displayRectangle.width()
        layout.minimumHeight = displayRectangle.height()

        setContentView(layout)

        bt_cancel_dialog_gls_bar_chart.setOnClickListener {
            cancel()
        }

        bt_ok_dialog_gls_bar_chart.setOnClickListener {
            dismiss()
        }



//        val lineDataSet1 = LineDataSet(getEntryList(0.1F), "c1")
//        val lineDataSet2 = LineDataSet(getEntryList(0.6F), "c2")


//        lineDataSet.setColors(*ColorTemplate.JOYFUL_COLORS)
//
//        lineDataSet.fillAlpha = 110

//        val lineData = LineData(lineDataSet1)

        val dataSets: ArrayList<ILineDataSet> = getDataSetLineChart()
        val dataLines = LineData(dataSets)

        lineChartGLS.getDescription().setText("Global Longitudinal Strain by frame")
        lineChartGLS.setData(dataLines)
//        lineChartGLS.setVisibleXRangeMaximum(10.0F)
        lineChartGLS.invalidate()


    }
    private val colors = intArrayOf(
        ColorTemplate.VORDIPLOM_COLORS[0],
        ColorTemplate.VORDIPLOM_COLORS[1],
        ColorTemplate.VORDIPLOM_COLORS[2],
        ColorTemplate.VORDIPLOM_COLORS[3],
        ColorTemplate.VORDIPLOM_COLORS[4],

        ColorTemplate.LIBERTY_COLORS[0],
        ColorTemplate.LIBERTY_COLORS[1],
        ColorTemplate.LIBERTY_COLORS[2],
        ColorTemplate.LIBERTY_COLORS[3],
        ColorTemplate.LIBERTY_COLORS[4],

        ColorTemplate.JOYFUL_COLORS[0],
        ColorTemplate.JOYFUL_COLORS[1],
        ColorTemplate.JOYFUL_COLORS[2],
        ColorTemplate.JOYFUL_COLORS[3],
        ColorTemplate.JOYFUL_COLORS[4],

        ColorTemplate.PASTEL_COLORS[0],
        ColorTemplate.PASTEL_COLORS[1],
        ColorTemplate.PASTEL_COLORS[2],
        ColorTemplate.PASTEL_COLORS[3],
        ColorTemplate.PASTEL_COLORS[4]
    )

//    fun getEntryList(r: Float): ArrayList<Entry> {
    fun getLineDataSet(r: Float): LineDataSet {
        val entryList = ArrayList<Entry>()
        entryList.add(Entry(10.0F * r, 20.0F))
        entryList.add(Entry(5.0F * r, 10.0F))
        entryList.add(Entry(7.0F * r, 31.0F))
        entryList.add(Entry(3.0F * r, 14.0F))

        val d = LineDataSet(entryList, "DataSet ${r}" )
        d.setLineWidth(2.5f)
        d.setCircleRadius(4f)
        val z = 0
        val color: Int = colors.get(z % colors.size)
        d.setColor(color)
        d.setCircleColor(color)
        return d
    }

    fun getDataSetLineChart(numFrame:Int=100, numSector: Int=6, randomSeed: Int=20): ArrayList<ILineDataSet> {
        val dataSets: ArrayList<ILineDataSet> = ArrayList()

        // random: https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.random/-random.html
        fun getRandomList(numFrame: Int, random: Random): List<Float> =
            List(numFrame) { random.nextDouble(-22.0, -14.0).toFloat() }

        repeat(numSector) {idSector ->

            val listY = getRandomList(numFrame, Random(idSector))
//            Log.w(TAG, "getDataSetLineChart ${numSector} ${listY.size}")
            val entryList = ArrayList<Entry>()

            repeat(numFrame) { idFrame ->
//                Log.w(TAG, "GET y: ${idFrame} ${listY.get(idFrame)}")
                entryList.add(Entry( (idFrame + 1).toFloat(), listY.get(idFrame)))
            }
//            Log.w(TAG, "GO AFTER add entry")

            val d = LineDataSet(entryList, "Sector ${idSector + 1}" )

            d.setLineWidth(2.5f)
            d.setCircleRadius(4f)

            val color: Int = colors.get(idSector % colors.size)

            d.setColor(color)
            d.setCircleColor(color)
            dataSets.add(d)

        }
        return dataSets
    }

}