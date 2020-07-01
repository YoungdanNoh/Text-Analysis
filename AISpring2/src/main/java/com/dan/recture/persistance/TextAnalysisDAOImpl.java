package com.dan.recture.persistance;

import java.util.List;

import javax.inject.Inject;

import org.apache.ibatis.session.SqlSession;
import org.springframework.stereotype.Repository;

import com.dan.recture.domain.TextAnalysisModel;

@Repository
public class TextAnalysisDAOImpl implements TextAnalysisDAO{

	@Inject
	private SqlSession session;
	
//	private static String namespace="com.dan.mappers.BoardMapper";
	private static String namespace="com.dan.mappers.AIMapper";
	
//	@Override
//	public List<TextAnalysisModel> list(TextAnalysisModel result)throws Exception{
//		return session.selectList(namespace+".list",result);
//	}
	@Override
	public List<TextAnalysisModel> Result() throws Exception{
		return session.selectList(namespace+".list");
	}
}
