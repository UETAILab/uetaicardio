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
import com.ailab.aicardiotrainer.repositories.SkillItem
import kotlinx.android.synthetic.main.item_skill.view.*

class SkillRVAdapter (
    val studyListener: OnStudyClicked
)
    : ListAdapter<SkillItem, SkillRVAdapter.ViewHolder>(NewsItemItemCallback()) {

    val adapterStore = HashMap<String, StudyRVAdapter>()

    inner class ViewHolder(val container: View) : RecyclerView.ViewHolder(container) {

        fun bind(item: SkillItem?, position: Int) {
            item?.let {
                itemView.tv_skill_name.text = item.name

                adapterStore.get(item.name)?.let {
                    itemView.rv_studies.apply { adapter = it }
                } ?: run {
                    val studyRVAdapter = StudyRVAdapter(item.name, studyListener)
                    adapterStore[item.name] = studyRVAdapter
                    itemView.rv_studies.apply { adapter = studyRVAdapter }
                    studyRVAdapter.submitList(item.studies)
                }
            }
        }

    }

    override fun getItemCount(): Int {
        return currentList.size
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        return ViewHolder(LayoutInflater.from(parent.context).inflate(R.layout.item_skill, parent, false))
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(getItem(position), position)
    }

    internal class NewsItemItemCallback : DiffUtil.ItemCallback<SkillItem>() {
        override fun areItemsTheSame(oldItem: SkillItem, newItem: SkillItem): Boolean {
            return oldItem == newItem
        }

        override fun areContentsTheSame(oldItem: SkillItem, newItem: SkillItem): Boolean {
            return oldItem.name == newItem.name
        }
    }
}
