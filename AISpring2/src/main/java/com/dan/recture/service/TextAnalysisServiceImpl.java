package com.dan.recture.service;

import java.util.List;

import javax.inject.Inject;

import org.springframework.stereotype.Repository;
import org.springframework.stereotype.Service;

import com.dan.recture.domain.TextAnalysisModel;
import com.dan.recture.persistance.TextAnalysisDAO;

@Service
public class TextAnalysisServiceImpl implements TextAnalysisService{

	@Inject
	private TextAnalysisDAO dao;
	
//	@Override
//	public List<TextAnalysisModel> list(TextAnalysisModel result)throws Exception{
//		return dao.list(result);
//	}
	@Override
	public List<TextAnalysisModel> Result()throws Exception{
		return dao.Result();
	}
}
