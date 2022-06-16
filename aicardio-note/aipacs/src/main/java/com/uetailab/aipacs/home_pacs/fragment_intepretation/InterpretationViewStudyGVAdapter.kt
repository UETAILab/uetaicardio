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

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import coil.api.load
import com.uetailab.aipacs.R
import kotlinx.android.synthetic.main.fragment_home_item_study_gv_preview.view.*

class InterpretationViewStudyGVAdapter(val applicationContext: Context, val listener: OnStudyPreviewClicked,
                                       val interpretationViewVM: InterpretationViewVM) : BaseAdapter() {

    companion object {
        const val TAG = "HomeViewStudyGVAdapter"
    }

    var items : List<StudyGVItem> = emptyList()

    override fun getView(position: Int, convertView: View?, parent: ViewGroup?): View {

        convertView?.let {
            getViewHolderFromTag(it).bind(getItem(position), getHasAnnotation(position))
            return it
        } ?: run {
            val view = LayoutInflater.from(applicationContext).inflate(R.layout.fragment_home_item_study_gv_preview, parent, false)
            ViewHolder(view).bind(
                getItem(position),
                getHasAnnotation(position)
            )
            return view
        }
    }

    fun getHasAnnotation(position: Int): Boolean {
        val relativePath = items.get(position).name
        return interpretationViewVM.getHasAnnotation(relativePath)

    }

    inner class ViewHolder(val itemView: View) {
        init {
            itemView.tag = this
        }

        fun bind(item: StudyGVItem, hasAnnotation: Boolean) {
            Log.w(TAG, item.name) // 1.2.840.113619.2.300.7348.1565874381.0.190____J8FGPA82
            val shortNameArr = item.name.split("____")
            val shortName = shortNameArr[shortNameArr.size - 1]
//
            itemView.tv_dicom_name.text = shortName
            itemView.iv_dicom_image.load(BitmapFactory.decodeFile(item.img_path))
            val dataView = interpretationViewVM.getDicomViewAndNumberOfFrame(shortName)
            itemView.tv_dicom_view.text = dataView.first
            itemView.tv_number_of_frame.text = dataView.second

//            itemView.
            itemView.ivHasAnnotation.visibility = if (hasAnnotation) View.VISIBLE else View.GONE

//            // Note here: file name length in android system is limited .2.840.113619.2.300.7348.1516976410.0.52____I1QF7NO2
//
            itemView.setOnClickListener {
                listener.onStudyPreviewClicked(item)
            }
            itemView.setOnLongClickListener {
                listener.onStudyPreviewLongClicked(item)
            }

        }
    }

    private fun getViewHolderFromTag(view: View): ViewHolder {
        return view.tag as ViewHolder
    }

    override fun getItem(position: Int): StudyGVItem {
        return items.get(position)
    }

    override fun getItemId(position: Int): Long {
        return getItem(position).id
    }

    override fun getCount(): Int {
        return items.size
    }

    fun submitList(newList : List<StudyGVItem>) {
        items = newList
        notifyDataSetChanged()
    }
    fun updateItem() {
//        items.forEach { item->
//            if (item.name == relativePath) {
//                bind(item, true)
//            }
//        }
        notifyDataSetChanged()
    }
}