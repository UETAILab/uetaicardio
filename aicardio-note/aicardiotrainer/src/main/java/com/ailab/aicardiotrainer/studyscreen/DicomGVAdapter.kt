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

package com.ailab.aicardiotrainer.studyscreen

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import coil.api.load
import com.ailab.aicardiotrainer.R
import com.ailab.aicardiotrainer.interfaces.OnDicomPreviewClicked
import kotlinx.android.synthetic.main.item_dicom_preview.view.*

class DicomGVAdapter(val applicationContext: Context, val listener: OnDicomPreviewClicked, val studyActVM: StudyActVM) : BaseAdapter() {
    companion object {
        const val TAG = "DicomGVAdapter"
    }

    var items : List<DicomItem> = emptyList()

    override fun getView(position: Int, convertView: View?, parent: ViewGroup?): View {

        convertView?.let {
            getViewHolderFromTag(it).bind(getItem(position))
            return it
        } ?: run {
            val view = LayoutInflater.from(applicationContext).inflate(R.layout.item_dicom_preview, parent, false)
            ViewHolder(view).bind(getItem(position))
            return view
        }
    }

    inner class ViewHolder(val itemView: View) {
        init {
            itemView.tag = this
        }

        fun bind(item: DicomItem) {
//            Log.w(TAG, item.name)
            val shortNameArr = item.name.split("____")
            val shortName = shortNameArr[shortNameArr.size - 1]

            itemView.tv_dicom_name.text = shortName
            itemView.iv_dicom_image.load(BitmapFactory.decodeFile(item.imgPath))
            val dataView = studyActVM.getDicomViewAndNumberOfFrame(shortName)
            itemView.tv_dicom_view.text = dataView.first
            itemView.tv_number_of_frame.text = dataView.second
            // Note here: file name length in android system is limited .2.840.113619.2.300.7348.1516976410.0.52____I1QF7NO2

            itemView.setOnClickListener {
                listener.onDicomPreviewClicked(item)
            }
            itemView.setOnLongClickListener {
                listener.onDicomPreviewLongClicked(item)
            }

        }
    }

    private fun getViewHolderFromTag(view: View): ViewHolder {
        return view.tag as ViewHolder
    }

    override fun getItem(position: Int): DicomItem {
        return items.get(position)
    }

    override fun getItemId(position: Int): Long {
        return getItem(position).id
    }

    override fun getCount(): Int {
        return items.size
    }

    fun submitList(newList : List<DicomItem>) {
        items = newList
        notifyDataSetChanged()
        Log.w(TAG, "submitList")
    }
}