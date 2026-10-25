import pytest
from scripts.report_frozen_review import selected_records,summarize_group


def test_followups_include_every_mode_and_omit_first_turn():
    rows=[{'record_id':f'{m}_{t}','turn':t,'mode':m} for m in ('none','full','compact') for t in (1,2,3,4)]
    picked=selected_records(rows,'conversation')
    assert len(picked)==9 and {r['mode'] for r in picked}=={'none','full','compact'}
    assert all(r['turn']>=2 for r in picked)


def test_repair_minimum_comparison_keeps_zero_and_two():
    rows=[{'record_id':str(m),'mode':str(m)} for m in range(3)]
    assert [r['mode'] for r in selected_records(rows,'repair')]==['0','2']


def test_incomplete_labels_do_not_report_partial_sample_accuracy():
    records=[{'record_id':str(n)} for n in range(3)]
    summary=summarize_group(records,{'0':{'answer_correct':'Y'}})
    assert summary['scored']==1 and summary['strict_accuracy'] is None
    full=summarize_group(records,{str(n):{'answer_correct':g} for n,g in enumerate(('Y','P','N'))})
    assert full['strict_accuracy']==1/3 and full['effective_answer_rate']==2/3


def test_invalid_grade_is_not_silently_counted_as_missing():
    with pytest.raises(ValueError):
        summarize_group([{'record_id':'x'}],{'x':{'answer_correct':'true'}})
