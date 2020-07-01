package com.dan.recture.persistance;

import java.util.List;

import javax.inject.Inject;

import org.apache.ibatis.session.SqlSession;
import org.springframework.stereotype.Repository;

import com.dan.recture.domain.AirbnbResultVO;


@Repository
public class AirbnbResultDAOImpl implements AirbnbResultDAO{
	
	@Inject
	SqlSession session;

	private static String namespace = "com.dan.mappers.AIMapper";
	
	@Override
	public List<AirbnbResultVO> selectResult() throws Exception {
		// TODO Auto-generated method stub
		
		return session.selectList(namespace+".selectResult");
	}

}
