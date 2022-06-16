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

package com.ailab.aicardiotrainer.nav_interpretation.ui.annotation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import coil.api.load
import com.ailab.aicardiotrainer.R
import kotlinx.android.synthetic.main.item_dicom_preview.view.*

class StudyRepresentationGVAdapter(val applicationContext: Context, val listener: OnSopInstanceUIDItemClicked) : BaseAdapter() {
    companion object {
        const val TAG = "StudyRepresentationGVAdapter"
    }

    private var currentPosition = -1

    fun setCurrentPosition(frameIdx: Int) {

//        Log.w(TAG, "setCurrentPosition: id: ${frameIdx} cur: ${currentPosition}")
//        if (currentPosition != -1 && currentPosition < count) notifyDataSetChanged(currentPosition)
//        currentPosition = frameIdx
//        notifyItemChanged(frameIdx)
    }

    fun getCurrentPosition() : Int {
        return currentPosition
    }

    var items : List<SopInstanceUIDItem> = emptyList()

    override fun getView(position: Int, convertView: View?, parent: ViewGroup?): View {

        convertView?.let {
            getViewHolderFromTag(it).bind(getItem(position), position)
            return it
        } ?: run {
            val view = LayoutInflater.from(applicationContext).inflate(R.layout.item_dicom_preview_small, parent, false)
            ViewHolder(view).bind(getItem(position), position)
            return view
        }
    }





    inner class ViewHolder(val itemView: View) {
        init {
            itemView.tag = this
        }

        fun bind(item: SopInstanceUIDItem, position: Int) {
//            itemView.
//            itemView.setBackgroundColor(if (position == currentPosition) containerView.context.getColor(R.color.app_blue_dark) else containerView.context.getColor(R.color.colorBackground))

            val shortNameArr = item.name.split("____")
            val shortName = shortNameArr[shortNameArr.size - 1]

            itemView.tv_dicom_name.text = shortName
//            val bm = getResizedBitmap(BitmapFactory.decodeFile(item.imgPath), 50, 35)

//            itemView.iv_dicom_image.load(getResizedBitmap(BitmapFactory.decodeFile(item.imgPath), 50, 35))

            itemView.iv_dicom_image.load(item.bitmap)
//            itemView.iv_dicom_image.load(item.bitmap)



            itemView.setOnClickListener {
//                Log.w(TAG, "onSopInstanceUIDItemClicked ${item.imgPath}")
                listener.onSopInstanceUIDItemClicked(item)
            }
            itemView.setOnLongClickListener {
//                Log.w(TAG, "onSopInstanceUIDItemLongClicked ${item.imgPath}")
                listener.onSopInstanceUIDItemLongClicked(item)
                true
            }

        }
    }

    private fun getViewHolderFromTag(view: View): ViewHolder {
        return view.tag as ViewHolder
    }

    override fun getItem(position: Int): SopInstanceUIDItem {
        return items.get(position)
    }

    override fun getItemId(position: Int): Long {
        return getItem(position).id
    }

    override fun getCount(): Int {
        return items.size
    }

    fun submitList(newList : List<SopInstanceUIDItem>) {
        items = newList
        notifyDataSetChanged()
//        Log.w(TAG, "submitList")
    }
}