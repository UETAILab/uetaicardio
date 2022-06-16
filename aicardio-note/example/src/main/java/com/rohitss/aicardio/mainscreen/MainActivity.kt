package com.rohitss.mvr.annotatescreen

import android.os.Bundle
import android.util.Log
import androidx.activity.viewModels
import com.google.android.material.snackbar.Snackbar
import com.rohitss.aacmvi.AacMviActivity
import com.rohitss.mvr.NewsRvAdapter
import com.rohitss.mvr.R
import com.rohitss.mvr.repository.NewsItem
import com.rohitss.mvr.toast
import kotlinx.android.synthetic.main.activity_main.*
import java.util.logging.Logger

class MainActivity : AacMviActivity<AnnotateViewState, AnnotateViewEffect, AnnotateViewEvent, MainActVM>() {
    override val viewModel: MainActVM by viewModels()

    companion object {
        val TAG: String = "MainActivity"
    }

    private val newsRvAdapter by lazy {
        NewsRvAdapter {
            viewModel.process(AnnotateViewEvent.NewsItemClicked(it.tag as NewsItem))
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        Log.w(TAG,"Message from onCreate")
        rvNewsHome.adapter = newsRvAdapter

        srlNewsHome.setOnRefreshListener {
            viewModel.process(AnnotateViewEvent.OnSwipeRefresh)
        }

        fabStar.setOnClickListener {
            viewModel.process(AnnotateViewEvent.FabClicked)
        }
    }

    override fun renderViewState(viewState: AnnotateViewState) {
        when (viewState.fetchStatus) {
            is FetchStatus.Fetched -> {
                srlNewsHome.isRefreshing = false
            }
            is FetchStatus.NotFetched -> {
                viewModel.process(AnnotateViewEvent.FetchNews)
                srlNewsHome.isRefreshing = false
            }
            is FetchStatus.Fetching -> {
                Log.w(TAG, "fetching")
                srlNewsHome.isRefreshing = true
            }
        }
        newsRvAdapter.submitList(viewState.newsList)
    }

    override fun renderViewEffect(viewEffect: AnnotateViewEffect) {
        when (viewEffect) {
            is AnnotateViewEffect.ShowSnackbar -> {
                Snackbar.make(coordinatorLayoutRoot, viewEffect.message, Snackbar.LENGTH_SHORT).show()
            }
            is AnnotateViewEffect.ShowToast -> {
                toast(message = viewEffect.message)
            }
        }
    }
}

