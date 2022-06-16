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

package com.ailab.aicardiotrainer.mainscreen

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.ailab.aicardiotrainer.R
import com.ailab.aicardiotrainer.interfaces.OnStudyClicked
import com.ailab.aicardiotrainer.repositories.StudyItem
import kotlinx.android.synthetic.main.item_study.view.*

class StudyRVAdapter(val name: String, val studyListener: OnStudyClicked) : ListAdapter<StudyItem, StudyRVAdapter.ViewHolder>(
    NewsItemItemCallback()
) {
    inner class ViewHolder(val container: View) : RecyclerView.ViewHolder(container) {
        init {
            itemView.setOnClickListener {
                studyListener.onStudyClicked(name, itemView.tag as StudyItem)
            }
        }

        fun bind(item: StudyItem?, position: Int) {
            item?.let {
                itemView.tag = it
                itemView.tv_study_name.text = it.name
                itemView.tv_dicom_count.text = it.dicomCount.toString()
//                itemView.iv_study_image.load(R.mipmap.ic_launcher_round)
            }
        }
    }

    override fun getItemCount(): Int {
        return currentList.size
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        return ViewHolder(LayoutInflater.from(parent.context).inflate(R.layout.item_study, parent, false))
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(getItem(position), position)
    }

    internal class NewsItemItemCallback : DiffUtil.ItemCallback<StudyItem>() {
        override fun areItemsTheSame(oldItem: StudyItem, newItem: StudyItem): Boolean {
            return oldItem == newItem
        }

        override fun areContentsTheSame(oldItem: StudyItem, newItem: StudyItem): Boolean {
            return oldItem.name == newItem.name
        }
    }
}
